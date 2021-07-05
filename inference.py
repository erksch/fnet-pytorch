import argparse

import torch
import sentencepiece as spm


def main(args):
    fnet = torch.load(args.model)
    fnet.eval()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.vocab)
    tokenizer.SetEncodeExtraOptions("")

    seq_len = fnet.config['max_position_embeddings']
    padded_input_ids = torch.full((1, seq_len), tokenizer.pad_id()).long()
    input_ids = torch.LongTensor(tokenizer.EncodeAsIds(args.text))
    padded_input_ids[0, :len(input_ids)] = input_ids

    sequence_output, pooled_output = fnet(padded_input_ids)

    print("Sequence output:", sequence_output.shape, sequence_output)
    print("Pooled output:", pooled_output.shape, pooled_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', '-m', type=str, required=True, help='path to PyTorch model (.pt)')
    parser.add_argument('--vocab', '-v', type=str, required=True, help='path to sentencepiece model')
    parser.add_argument('--text', '-t', type=str, default="Hello welcome to the show", help='input text')

    main(parser.parse_args())
