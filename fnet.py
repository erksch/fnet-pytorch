import torch
import torch.utils.checkpoint
from torch import nn


class FNetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'],
                                            padding_idx=config['pad_token_id'])
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])

        self.layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.hidden_mapping = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

        self.position_ids = torch.arange(config['max_position_embeddings']).expand((1, -1))

    def forward(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]
        token_type_ids = torch.zeros(input_shape, dtype=torch.long)

        word_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
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

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids=input_ids)
        sequence_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output
