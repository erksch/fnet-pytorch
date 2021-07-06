# FNet PyTorch

A PyTorch implementation of FNet from the paper _FNet: Mixing Tokens with Fourier Transforms_ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon ([arXiv](https://arxiv.org/abs/2105.03824)).

Additional to the architecture implementation, this repository offers a script for converting a checkpoint from the [official FNet implementation](https://github.com/google-research/google-research/tree/master/f_net) (written in Jax) to a PyTorch checkpoint (statedict and or model export).

## Setup

Install dependencies (ideally in a virtualenv)

```bash
pip install -r requirements.txt
```

## Using a pre-trained model

You can download a pre-trained model from [here](https://drive.google.com/file/d/1HAHEIZDT70gz6hI_u3fN6jD6Rim5rq18/view?usp=sharing).
The pre-trained model was obtained by converting the official Jax checkpoint of FNet-Base using the technique described below.

You have two options on how to load the pre-trained model:

Option 1: `config.json` & `fnet.statedict.pt`

```python
with open('path/to/config.json', 'r') as f:
    config = json.load(f)
fnet = FNet(config)
fnet.load_state_dict(torch.load('path/to/fnet.statedict.pt'))
```

Option 2: `fnet.pt`

```python
fnet = torch.load('path/to/fnet.pt')
```

## Jax checkpoint conversion

Download a pre-trained Jax checkpoint of FNet from their [official GitHub page](https://github.com/google-research/google-research/tree/master/f_net#base-models) or use any checkpoint that you trained using the official implementation.  
You also need the SentencePiece vocab model. For the official checkpoints, use the model given [here](https://github.com/google-research/google-research/tree/master/f_net#how-to-pre-train-or-fine-tune-fnet). For custom checkpoints use your respective vocab model.

To convert the checkpoint to PyTorch run

```bash
python convert_jax_checkpoint.py \
    --checkpoint <path/to/checkout> \
    --vocab <path/to/vocab> \
    --outdir <outdir>
```

This will save a `config.json`, `fnet.statedict.pt` and `fnet.pt` to `<outdir>`.

#### Disclaimer

Although all model parameters will be correctly transferred to the PyTorch model, there will be slight differences between Jax and PyTorch in the inference result because their LayerNorm and GELU implementations slightly differ.

If you run an example input through both models and compare e.g. the final hidden states they will not be entirely equal. But every float value will be equal up to the 4th digit after the comma (worst case! some float values will be more precisely equal).

Speaking programmatically, if `a` is the Jax FNet last hidden state and `b` is the PyTorch FNet last hidden state of the same input sample or batch `numpy.allclose(a, b, atol=1e-04)` will be `true`.
