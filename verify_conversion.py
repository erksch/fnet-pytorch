import argparse
import json

import ml_collections
import sentencepiece as spm

import torch
from flax.training import checkpoints
from jax import random
import jax
import jax.numpy as jnp
import numpy as np

from fnet import FNetForPreTraining
from f_net.models import PreTrainingModel as JaxPreTrainingModel
from f_net.configs.pretraining import get_config

def compare_output(jax_checkpoint_path, torch_statedict_path, torch_config_path, vocab_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(vocab_path)
    tokenizer.SetEncodeExtraOptions("")

    print("Loading PyTorch checkpoint...")

    with open(torch_config_path) as f:
        fnet_torch_config = json.load(f)
    fnet_torch = FNetForPreTraining(fnet_torch_config)
    statedict = torch.load(torch_statedict_path, map_location=torch.device('cpu'))
    fnet_torch.load_state_dict(statedict)
    fnet_torch.eval()

    print("Done")

    print("Loading Jax checkpoint...")

    random_seed = 0
    rng = random.PRNGKey(random_seed)
    rng, init_rng = random.split(rng)
    config = get_config()
    with config.unlocked():
        config.vocab_size = tokenizer.GetPieceSize()

    frozen_config = ml_collections.FrozenConfigDict(config)
    fnet_jax_model = JaxPreTrainingModel(config=frozen_config, random_seed=random_seed)
    fnet_jax_params = jax_init_params(fnet_jax_model, init_rng, frozen_config)
    fnet_jax_params = checkpoints.restore_checkpoint(jax_checkpoint_path, {'target': fnet_jax_params})['target']

    print("Done")

    input_ids, token_type_ids, mlm_positions, mlm_ids = get_input(tokenizer, fnet_torch_config['max_position_embeddings'])

    with torch.no_grad():
        fnet_torch_output = fnet_torch(input_ids, token_type_ids, mlm_positions)

    print(fnet_torch_output)

    fnet_jax_output = fnet_jax_model.apply({"params": fnet_jax_params}, **{
        "input_ids": input_ids.numpy(),
        "input_mask": (input_ids.numpy() > 0).astype(np.int32),
        "type_ids": token_type_ids.numpy(),
        "masked_lm_positions": mlm_positions.numpy(),
        "masked_lm_labels": mlm_ids.numpy(),
        "masked_lm_weights": (mlm_positions.numpy() > 0).astype(np.float32),
        "next_sentence_labels": np.array([1]),
        "deterministic": True
    })

    print(fnet_jax_output)

    atol = 1e-01

    assert np.allclose(fnet_torch_output['mlm_logits'].numpy(), fnet_jax_output['masked_lm_logits'], atol=atol)
    assert np.allclose(fnet_torch_output['nsp_logits'].numpy(), fnet_jax_output['next_sentence_logits'], atol=atol)

    print(f"Inference results of both models are equal up to {atol}")

def jax_init_params(model, key, config):
  init_batch = {
      "input_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "input_mask": jnp.ones((1, config.max_seq_length), jnp.int32),
      "type_ids": jnp.ones((1, config.max_seq_length), jnp.int32),
      "masked_lm_positions": jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "masked_lm_labels": jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "masked_lm_weights": jnp.ones((1, config.max_predictions_per_seq), jnp.int32),
      "next_sentence_labels": jnp.ones((1, 1), jnp.int32)
  }

  key, dropout_key = random.split(key)

  jit_init = jax.jit(model.init)
  initial_variables = jit_init({
      "params": key,
      "dropout": dropout_key
  }, **init_batch)

  return initial_variables["params"]

def get_input(tokenizer, seq_len):
    text = "Joseph Harold Greenberg (May 28, 1915 – May 7, 2001) was an American linguist, " \
           "known mainly for his work concerning " \
           "linguistic typology and the genetic classification of languages."

    cls_id = tokenizer.PieceToId("[CLS]")
    mask_id = tokenizer.PieceToId("[MASK]")
    sep_id = tokenizer.PieceToId("[SEP]")
    pad_id = tokenizer.pad_id()

    token_ids = [cls_id] + tokenizer.EncodeAsIds(text) + [sep_id]
    input_ids = torch.full((1, seq_len), pad_id, dtype=torch.long)
    input_ids[0, :len(token_ids)] = torch.LongTensor(token_ids)

    # mask some tokens
    mlm_positions = torch.LongTensor([1, 5, 7])
    mlm_ids = input_ids[0, mlm_positions]
    input_ids[0, mlm_positions] = mask_id

    token_type_ids = torch.full((1, seq_len), 0, dtype=torch.long)

    max_mlm_maskings = 80
    full_mlm_positions = torch.full((1, max_mlm_maskings), 0, dtype=torch.long)
    full_mlm_positions[:, :len(mlm_positions)] = mlm_positions

    full_mlm_ids = torch.full((1, max_mlm_maskings), 0, dtype=torch.long)
    full_mlm_ids[:, :len(mlm_ids)] = mlm_ids

    return input_ids, token_type_ids, full_mlm_positions, full_mlm_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--jax', type=str, required=True, help='path to FNet jax checkpoint')
    parser.add_argument('--torch', type=str, required=True, help='path to PyTorch statedict checkpoint')
    parser.add_argument('--config', type=str, required=True, help='path to PyTorch checkpoint config')
    parser.add_argument('--vocab', type=str, required=True, help='path to vocab file')

    args = parser.parse_args()

    compare_output(args.jax, args.torch, args.config, args.vocab)
