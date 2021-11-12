from abc import ABC, abstractmethod
from typing import List, Dict
import torch
import sentencepiece as spm

from typing import TypedDict

from tokenizers import Tokenizer as HFTokenizer, BertWordPieceTokenizer


class EncodedText(TypedDict):
    input_ids: torch.Tensor
    type_ids: torch.Tensor


class Tokenizer(ABC):
    pad_id: int
    sep_id: int
    mask_id: int
    cls_id: int
    vocab_size: int

    def special_tokens(self) -> List[int]:
        return [self.mask_id, self.cls_id, self.sep_id, self.pad_id]

    @abstractmethod
    def encode_as_ids(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        pass

    @abstractmethod
    def encode(self, texts: List[str]) -> EncodedText:
        pass


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer

        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def encode_as_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, is_pretokenized=False, add_special_tokens=False).ids

    def encode(self, texts: List[str]) -> EncodedText:
        if len(texts) > 2:
            raise Exception("Hugging face tokenizer can only encode two texts")
        elif len(texts) == 2:
            sequence, pair = texts
        else:
            sequence, pair = texts[0], None

        encoding = self.tokenizer.encode(sequence, pair)

        return {"input_ids": encoding.ids, "type_ids": encoding.type_ids}


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, vocab_file, max_seq_length):
        super(SentencePieceTokenizer).__init__()

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(vocab_file)
        self.tokenizer.SetEncodeExtraOptions("")

        self.vocab_size = self.tokenizer.GetPieceSize()

        self.cls_id = self.tokenizer.PieceToId("[CLS]")
        self.sep_id = self.tokenizer.PieceToId("[SEP]")
        self.mask_id = self.tokenizer.PieceToId("[MASK]")
        self.pad_id = self.tokenizer.pad_id()

        self.max_seq_length = max_seq_length

    def special_tokens(self):
        eos_id = self.tokenizer.eos_id()
        bos_id = self.tokenizer.bos_id()
        return {self.mask_id, self.cls_id, self.sep_id, self.pad_id, bos_id, eos_id}

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.DecodeIdsWithCheck(ids)

    def encode_as_ids(self, text: str) -> List[int]:
        return self.tokenizer.EncodeAsIds(text)

    def encode(self, texts: List[str]) -> EncodedText:
        input_ids_out = torch.full([self.max_seq_length], self.pad_id, dtype=torch.long)
        type_ids_out = torch.full([self.max_seq_length], 0, dtype=torch.long)

        input_ids = [self.cls_id]
        type_ids = [0]

        for text in texts:
            tokens = self.tokenizer.EncodeAsIds(text) + [self.sep_id]
            input_ids.extend(tokens)
            type_ids.extend([1] * len(tokens))

        # truncate
        input_ids = input_ids[:self.max_seq_length]
        type_ids = type_ids[:self.max_seq_length]

        # pad
        input_ids_out[:len(input_ids)] = torch.LongTensor(input_ids)
        type_ids_out[:len(type_ids)] = torch.LongTensor(type_ids)

        return {'input_ids': input_ids_out, 'type_ids': type_ids_out}


def get_tokenizer(tokenizer_config: Dict, max_seq_length: int):
    type = tokenizer_config['type']

    if type == 'sentencepiece':
        if not tokenizer_config['vocab']: raise Exception("No vocab given")
        return SentencePieceTokenizer(tokenizer_config['vocab'], max_seq_length)

    if type == 'wordpiece':
        if not tokenizer_config['vocab']: raise Exception("No vocab given")
        tokenizer = BertWordPieceTokenizer.from_file(tokenizer_config['vocab'])
        return HuggingFaceTokenizer(tokenizer)

    if type == 'huggingface':
        if not tokenizer_config['hf_name']: raise Exception("No name given")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config['hf_name'])
        return HuggingFaceTokenizer(tokenizer._tokenizer)

    raise Exception(f"Unexpected tokenizer type {type}")