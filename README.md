# FNet PyTorch

A PyTorch implementation of FNet from the paper _FNet: Mixing Tokens with Fourier Transforms_ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon ([arXiv](https://arxiv.org/abs/2105.03824)).

Additional to the architecture implementation, this repository offers a script for converting a checkpoint from the [official FNet implementation](https://github.com/google-research/google-research/tree/master/f_net) (written in Jax) to a PyTorch checkpoint (statedict and or model export).


## Using a pre-trained model

We offer the following converted checkpoints and pre-trained models

| Model | Jax checkpoint | PyTorch checkpoint | Arch Info | Dataset | Train Info |
| ----- | ---------------| ------------------- | --- | ---- | ---- |
| FNet Base | [checkpoint (official)](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/f_net_checkpoint) | [checkpoint (converted)](TODO) | E 768, D 768, FF 3072, 12 layers | C4 | see paper / official project |
| FNet Large | [checkpoint (official)](https://storage.googleapis.com/gresearch/f_net/checkpoints/large/f_net_checkpoint) | [checkpoint (converted)](TODO) | E 1024, D 1024, FF 4096, 24 layers | C4 | see paper / official project |
| FNet Small | [checkpoint (ours)](TODO) | [encoder checkpoint (converted)](TODO), [pretraining model checkpoint (converted)](TODO) | E 312, D 312, FF 3072, 4 layers | Wikipedia EN | trained with official training code. 1M steps, BS 64, LR 1e-4 |

The PyTorch checkpoints marked with *converted* are converted Jax checkpoints using the technique described below.

You can install this repository as a package running

```python
pip install git+https://github.com/erksch/fnet-pytorch
```

Now, you can load a pre-trained model in PyTorch as follows. 
You'll need the `config.json` and the `.statedict.pt` file.

```python
import torch
import json
from fnet import FNet, FNetForPretraining

with open('path/to/config.json', 'r') as f:
    config = json.load(f)

# if you just want the encoder
fnet = FNet(config)
fnet.load_state_dict(torch.load('path/to/fnet.statedict.pt'))

# if you want FNet with pre-training head
fnet = FNetForPretraining(config)
fnet.load_state_dict(torch.load('path/to/fnet_pretraining.statedict.pt'))
```

## Jax checkpoint conversion

Download a pre-trained Jax checkpoint of FNet from their [official GitHub page](https://github.com/google-research/google-research/tree/master/f_net#base-models) or use any checkpoint that you trained using the official implementation.  
You also need the SentencePiece vocab model. For the official checkpoints, use the model given [here](https://github.com/google-research/google-research/tree/master/f_net#how-to-pre-train-or-fine-tune-fnet). For custom checkpoints use your respective vocab model.

Install dependencies (ideally in a virtualenv)

```bash
pip install -r requirements.txt
```

Convert a checkpoint to PyTorch

```bash
python convert_jax_checkpoint.py \
    --checkpoint <path/to/checkout> \
    --vocab <path/to/vocab> \
    --outdir <outdir>
```

Output files: `config.json`, `fnet.statedict.pt`, `fnet_pretraining.statedict.pt`

The checkpoints from the official Jax implementation are of complete pre-training models, meaning they contain encoder and pre-training head weights. 
The conversion will convert the Jax checkpoint to a PyTorch `statedict` of this project's `FNet` module (`fnet.statedict.pt`) and `FNetForPreTraining` module (`fnet_pretraining.statedict.pt`). 
You can use the model type for your needs whether you want to run further pre-trainings or not. 

#### Disclaimer

Although all model parameters will be correctly transferred to the PyTorch model, there will be slight differences between Jax and PyTorch in the inference result because their LayerNorm and GELU implementations slightly differ.

If you run an example input through both models and compare e.g. the final hidden states they will not be entirely equal. But every float value will be equal up to the 4th digit after the comma (worst case! some float values will be more precisely equal).

Speaking programmatically, if `a` is the Jax FNet last hidden state and `b` is the PyTorch FNet last hidden state of the same input sample or batch `numpy.allclose(a, b, atol=1e-04)` will be `true`.
