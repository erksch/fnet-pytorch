import re
import torch
import torch.utils.checkpoint
from scipy import linalg
from torch import nn


class FNetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['embedding_size'],
                                            padding_idx=config['pad_token_id'])
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['embedding_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['embedding_size'])

        self.layer_norm = nn.LayerNorm(config['embedding_size'], eps=config['layer_norm_eps'])
        self.hidden_mapping = nn.Linear(config['embedding_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout_rate'])

        self.register_buffer(
            'position_ids',
            torch.arange(config['max_position_embeddings']).expand((1, -1)),
            persistent=False
        )

    def forward(self, input_ids, type_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        word_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.hidden_mapping(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class FourierMMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dft_mat_seq = torch.tensor(linalg.dft(config['max_position_embeddings']))
        self.dft_mat_hidden = torch.tensor(linalg.dft(config['hidden_size']))

    def forward(self, hidden_states):
        hidden_states_complex = hidden_states.type(torch.complex128)
        return torch.einsum(
            "...ij,...jk,...ni->...nk",
            hidden_states_complex,
            self.dft_mat_hidden,
            self.dft_mat_seq
        ).real.type(torch.float32)


class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft = FourierMMLayer(config) if config['fourier'] == 'matmul' else FourierFFTLayer()
        self.mixing_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.feed_forward = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.output_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output


class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FNetLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class FNetPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)
        self.pooler = FNetPooler(config)

    def forward(self, input_ids, type_ids):
        embedding_output = self.embeddings(input_ids, type_ids)
        sequence_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class FNetForPreTraining(nn.Module):
    def __init__(self, config):
        super(FNetForPreTraining, self).__init__()
        self.encoder = FNet(config)

        self.embedding_size = config['embedding_size']
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_hidden_layers']

        self.mlm_intermediate = nn.Linear(self.hidden_size, self.embedding_size)
        self.activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(self.embedding_size)
        self.mlm_output = nn.Linear(self.embedding_size, self.vocab_size)

        self.nsp_output = nn.Linear(self.hidden_size, 2)

    def _mlm(self, x):
        x = self.mlm_intermediate(x)
        x = self.activation(x)
        x = self.mlm_layer_norm(x)
        x = self.mlm_output(x)
        return x

    def forward(self, input_ids, type_ids, mlm_positions=None):
        sequence_output, pooled_output = self.encoder(input_ids, type_ids)

        if mlm_positions is not None:
            mlm_input = sequence_output.take_along_dim(mlm_positions.unsqueeze(-1), dim=1)
        else:
            mlm_input = sequence_output

        mlm_logits = self._mlm(mlm_input)
        nsp_logits = self.nsp_output(pooled_output)
        return {"mlm_logits": mlm_logits, "nsp_logits": nsp_logits}


def get_config_from_statedict(state_dict,
                              fourier_type="fft",
                              pad_token_id=0,
                              layer_norm_eps=1e-12,
                              dropout_rate=0.1):
    is_pretraining_checkpoint = 'mlm_output.weight' in state_dict.keys()
    
    def prepare(key):
        if is_pretraining_checkpoint: 
            return f"encoder.{key}"
        return key

    regex = re.compile(prepare(r'encoder.layer.\d+.feed_forward.weight'))
    num_layers = len([key for key in state_dict.keys() if regex.search(key)])

    return {
        "num_hidden_layers": num_layers,
        "vocab_size": state_dict[prepare('embeddings.word_embeddings.weight')].shape[0],
        "embedding_size": state_dict[prepare('embeddings.word_embeddings.weight')].shape[1],
        "hidden_size": state_dict[prepare('encoder.layer.0.output_dense.weight')].shape[0],
        "intermediate_size": state_dict[prepare('encoder.layer.0.feed_forward.weight')].shape[0],
        "max_position_embeddings": state_dict[prepare('embeddings.position_embeddings.weight')].shape[0],
        "type_vocab_size": state_dict[prepare('embeddings.token_type_embeddings.weight')].shape[0],
        # the following parameters can not be inferred from the state dict and must be given manually
        "fourier": fourier_type,
        "pad_token_id": pad_token_id,
        "layer_norm_eps": layer_norm_eps,
        "dropout_rate": dropout_rate,
    }
