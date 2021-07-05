import os

import torch
import argparse
import sentencepiece as spm
from flax import serialization
from tensorflow.io import gfile

from fnet import FNet


def load_jax_checkpoint(path):
    with gfile.GFile(path, 'rb') as fp:
        checkpoint_contents = fp.read()
        return serialization.msgpack_restore(checkpoint_contents)


def to_torch(arr):
    return torch.Tensor(arr.copy())


def convert(target, jax_tree):
    jax_fnet_encoder = jax_tree['target']['encoder']
    jax_fnet_embedder = jax_fnet_encoder['embedder']

    target.embeddings.word_embeddings.weight.data = to_torch(jax_fnet_embedder['word']['embedding'])
    target.embeddings.position_embeddings.weight.data = to_torch(jax_fnet_embedder['position']['embedding'][0])
    target.embeddings.token_type_embeddings.weight.data = to_torch(jax_fnet_embedder['type']['embedding'])

    target.embeddings.layer_norm.bias.data = to_torch(jax_fnet_embedder['layer_norm']['bias'])
    target.embeddings.layer_norm.weight.data = to_torch(jax_fnet_embedder['layer_norm']['scale'])

    target.embeddings.hidden_mapping.weight.data = to_torch(jax_fnet_embedder['hidden_mapping_in']['kernel'].T)
    target.embeddings.hidden_mapping.bias.data = to_torch(jax_fnet_embedder['hidden_mapping_in']['bias'])

    # encoder layer
    for i in range(len(target.encoder.layer)):
        jax_fnet_encoder_layer = jax_fnet_encoder[f'encoder_{i}']
        jax_fnet_feed_forward = jax_fnet_encoder[f'feed_forward_{i}']

        target.encoder.layer[i].mixing_layer_norm.weight.data = to_torch(jax_fnet_encoder_layer['mixing_layer_norm']['scale'])
        target.encoder.layer[i].mixing_layer_norm.bias.data = to_torch(jax_fnet_encoder_layer['mixing_layer_norm']['bias'])

        target.encoder.layer[i].feed_forward.weight.data = to_torch(jax_fnet_feed_forward['intermediate']['kernel'].T)
        target.encoder.layer[i].feed_forward.bias.data = to_torch(jax_fnet_feed_forward['intermediate']['bias'])

        target.encoder.layer[i].output_dense.weight.data = to_torch(jax_fnet_feed_forward['output']['kernel'].T)
        target.encoder.layer[i].output_dense.bias.data = to_torch(jax_fnet_feed_forward['output']['bias'])

        target.encoder.layer[i].output_layer_norm.weight.data = to_torch(jax_fnet_encoder_layer['output_layer_norm']['scale'])
        target.encoder.layer[i].output_layer_norm.bias.data = to_torch(jax_fnet_encoder_layer['output_layer_norm']['bias'])

    # pooler
    target.pooler.dense.weight.data = to_torch(jax_fnet_encoder['pooler']['kernel'].T)
    target.pooler.dense.bias.data = to_torch(jax_fnet_encoder['pooler']['bias'])

    return target


def main(args):
    jax_tree = load_jax_checkpoint(args.checkpoint)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.vocab)
    tokenizer.SetEncodeExtraOptions("")

    encoder = jax_tree['target']['encoder']
    num_layers = len([key for key in encoder.keys() if "encoder_" in key])

    config = {
        'vocab_size': tokenizer.vocab_size(),
        'hidden_size': encoder['feed_forward_0']['output']['bias'].shape[0],
        'intermediate_size': encoder['feed_forward_0']['intermediate']['bias'].shape[0],
        'max_position_embeddings': encoder['embedder']['position']['embedding'].shape[1],
        'pad_token_id': tokenizer.pad_id(),
        'type_vocab_size': 4,
        # https://github.com/google-research/google-research/blob/master/f_net/models.py#L43
        'layer_norm_eps': 1e-12,
        'hidden_dropout_prob': 0.1,
        'num_hidden_layers': num_layers
    }

    print("Extracted config:", config)

    print("Converting Jax checkpoint...")

    target = FNet(config)
    target = convert(target, jax_tree)

    print("Done.")

    torch.save(target.state_dict(), os.path.join(args.outdir, "fnet.statedict.pt"))
    torch.save(target, os.path.join(args.outdir, "fnet.pt"))

    print(f"Saved PyTorch files to {args.outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='path to FNet jax checkpoint')
    parser.add_argument('--vocab', '-v', type=str, required=True, help='path to sentencepiece model')
    parser.add_argument('--outdir', '-o', type=str, required=True, help='dir where to save pytorch exports')

    main(parser.parse_args())
