import torch
import torch.utils.checkpoint
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
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

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


class FourierLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states, dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft = FourierLayer()
        self.mixing_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.feed_forward = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.output_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
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
        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']

        self.mlm_intermediate = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(hidden_size)
        self.mlm_output = nn.Linear(hidden_size, vocab_size)

        self.nsp_output = nn.Linear(hidden_size, 2)

    def _mlm(self, x):
        x = self.mlm_intermediate(x)
        x = self.activation(x)
        x = self.mlm_layer_norm(x)
        x = self.mlm_output(x)
        return x

    def forward(self, input_ids, type_ids, mlm_positions):
        sequence_output, pooled_output = self.encoder(input_ids, type_ids)
        mlm_input = sequence_output.take_along_dim(mlm_positions.unsqueeze(-1), dim=1)
        mlm_logits = self._mlm(mlm_input)
        nsp_logits = self.nsp_output(pooled_output)
        return {"mlm_logits": mlm_logits, "nsp_logits": nsp_logits}
