"""
Heavily inspired by https://github.com/google-research/google-research/blob/master/f_net/input_pipeline.py
"""
from typing import Iterator, List, TypedDict

import tensorflow_datasets as tfds
import numpy as np
import torch
import json
import glob

from .tokenization import Tokenizer

np.random.seed(0)


class NSPData(TypedDict):
    input_ids: torch.Tensor
    type_ids: torch.Tensor
    nsp_labels: int


class PreTrainData(TypedDict):
    input_ids: torch.Tensor
    original_input_ids: torch.Tensor
    type_ids: torch.Tensor
    mlm_positions: torch.Tensor
    mlm_ids: torch.Tensor
    mlm_weights: torch.Tensor
    nsp_labels: int


def pretraining_data_gen(
        tokenizer: Tokenizer,
        batch_size: int,
        max_seq_length: int,
        device: torch.device,
        max_predictions_per_seq=80,
        masking_rate=0.15,
        mask_token_proportion=0.8,
        random_token_proportion=0.1
) -> Iterator[PreTrainData]:
    ignore_ids = [tokenizer.cls_id, tokenizer.sep_id, tokenizer.pad_id]
    ignore_ids = torch.LongTensor(ignore_ids)[:, None]

    normal_tokens = [t for t in range(tokenizer.vocab_size) if t not in tokenizer.special_tokens()]

    gen = _nsp_data_gen(tokenizer, max_seq_length)

    samples = []
    for sample in gen:
        sample: PreTrainData = sample
        num_tokens = torch.sum(sample["input_ids"] != tokenizer.pad_id).item()
        prediction_mask = torch.all(sample["input_ids"] != ignore_ids, dim=0)
        cand_indices = torch.arange(prediction_mask.shape[0], dtype=torch.long)[prediction_mask]
        num_to_predict = min(max_predictions_per_seq, max(1, int(num_tokens * masking_rate)))

        if len(cand_indices) == 0:
            continue

        mlm_positions = torch.LongTensor(np.sort(np.random.choice(cand_indices, num_to_predict, replace=False)))
        mlm_ids = sample["input_ids"][mlm_positions]
        mlm_weights = torch.ones(num_to_predict, dtype=torch.float32)

        # Mask out tokens
        for position in mlm_positions:
            rand = np.random.random()
            if rand < mask_token_proportion:
                replace_token_id = tokenizer.mask_id
            elif rand < mask_token_proportion + random_token_proportion:
                replace_token_id = np.random.choice(normal_tokens, 1).item()
            else:
                replace_token_id = sample["input_ids"][position]
            sample["input_ids"][position] = replace_token_id

        mlm_positions_out = torch.zeros(max_predictions_per_seq, dtype=torch.long)
        mlm_ids_out = torch.zeros(max_predictions_per_seq, dtype=torch.long)
        mlm_weights_out = torch.zeros(max_predictions_per_seq, dtype=torch.float32)

        mlm_weights_out[:num_to_predict] = mlm_weights
        mlm_positions_out[:num_to_predict] = mlm_positions
        mlm_ids_out[:num_to_predict] = mlm_ids

        sample["mlm_positions"] = mlm_positions_out
        sample["mlm_ids"] = mlm_ids_out
        sample["mlm_weights"] = mlm_weights_out

        samples.append(sample)

        if len(samples) == batch_size:
            yield samples_to_batch(samples, device)
            samples = []


def _nsp_data_gen(
        tokenizer: Tokenizer,
        max_seq_length: int
) -> Iterator[NSPData]:
    ds = tfds.load(name='wikipedia/20201201.en', split="train", shuffle_files=True)
    ds = ds.repeat()
    ds = ds.shuffle(1024)
    ds = ds.batch(16)

    for batch in tfds.as_numpy(ds):
        for text in batch["text"]:
            text = str(text, "utf-8")
            lines = [tokenizer.encode_as_ids(line) for line in text.splitlines()]
            j = 0
            while j < len(lines) - 1:
                if len(lines[j]) + len(lines[j + 1]) > max_seq_length - 3:
                    j += 1
                    continue

                input_ids = torch.full((max_seq_length,), tokenizer.pad_id, dtype=torch.long)
                type_ids = torch.full((max_seq_length,), 1, dtype=torch.long)

                selected_lines = concat_lines_until_max(lines[j:], max_seq_length)
                j += len(selected_lines)

                pivot = np.random.randint(1, len(selected_lines))
                datum = [tokenizer.cls_id]

                if np.random.random() < 0.5:
                    for tokens in selected_lines[:pivot]:
                        datum.extend(tokens)
                    datum.append(tokenizer.sep_id)
                    type_ids[:len(datum)] = 0
                    for tokens in selected_lines[pivot:]:
                        datum.extend(tokens)
                    datum.append(tokenizer.sep_id)
                    next_sentence_label = 0
                    type_ids[len(datum):] = 0
                else:
                    for tokens in selected_lines[pivot:]:
                        datum.extend(tokens)
                    datum.append(tokenizer.sep_id)
                    type_ids[:len(datum)] = 0
                    for tokens in selected_lines[:pivot]:
                        datum.extend(tokens)
                    datum.append(tokenizer.sep_id)
                    next_sentence_label = 1
                    type_ids[len(datum):] = 0

                input_ids[:] = tokenizer.pad_id
                input_ids[:len(datum)] = torch.LongTensor(datum)

                yield {
                    "input_ids": input_ids,
                    "type_ids": type_ids,
                    "nsp_labels": next_sentence_label,
                }


def concat_lines_until_max(lines, max_len):
    cum_len = 0
    k = 0
    for k in range(len(lines)):
        cum_len += len(lines[k])
        if cum_len > max_len - 3:
            k -= 1
            break
    return lines[:k + 1]


def samples_to_batch(samples, device):
    batch_size = len(samples)
    batch = {}
    for key in samples[0].keys():
        value = samples[0][key]
        if isinstance(value, torch.Tensor):
            batch[key] = torch.zeros((batch_size, value.shape[0]), dtype=value.dtype).to(device)
        else:
            batch[key] = torch.zeros(batch_size, dtype=(torch.long if isinstance(value, int) else torch.float32)).to(device)
    for i, sample in enumerate(samples):
        for key in batch.keys():
            batch[key][i] = sample[key]
    return batch
