# FNet PyTorch

A PyTorch implementation of FNet from the paper _FNet: Mixing Tokens with Fourier Transforms_ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon ([arXiv](https://arxiv.org/abs/2105.03824)).

Additional to the architecture implementation, this repository offers a script for converting a checkpoint from the [official FNet implementation](https://github.com/google-research/google-research/tree/master/f_net) (written in Jax) to a PyTorch checkpoint (statedict and or model export).


## Using a pre-trained model

We offer the following converted checkpoints and pre-trained models

| Model | Jax checkpoint | PyTorch checkpoint | Arch Info | Dataset | Train Info |
| ----- | ---------------| ------------------- | --- | ---- | ---- |
| FNet Large | [checkpoint (official)](https://storage.googleapis.com/gresearch/f_net/checkpoints/large/f_net_checkpoint) | [checkpoint (converted)](https://voize-checkpoints-public.s3.eu-central-1.amazonaws.com/fnet/pytorch_checkpoints/fnet_large_pt_checkpoint.zip) | E 1024, D 1024, FF 4096, 24 layers | C4 | see paper / official project |
| FNet Base | [checkpoint (official)](https://storage.googleapis.com/gresearch/f_net/checkpoints/base/f_net_checkpoint) | [checkpoint (converted)](https://voize-checkpoints-public.s3.eu-central-1.amazonaws.com/fnet/pytorch_checkpoints/fnet_base_pt_checkpoint.zip) | E 768, D 768, FF 3072, 12 layers | C4 | see paper / official project |
| FNet Small | [checkpoint (ours)](https://voize-checkpoints-public.s3.eu-central-1.amazonaws.com/fnet/jax_checkpoints/fnet_small_jax_checkpoint) | [checkpoint (converted)](https://voize-checkpoints-public.s3.eu-central-1.amazonaws.com/fnet/pytorch_checkpoints/fnet_small_pt_checkpoint.zip) | E 768, D 312, FF 3072, 4 layers | Wikipedia EN | trained with official training code. 1M steps, BS 64, LR 1e-4 |

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

For a given inference input, all hidden states and logits of the official and converted model are equal at least up the first digit after the comma. This is programmatically verified using the script described below.

### Verify conversion results

You can use the `verify_conversion.py` script to compare the inference outputs of a Jax checkpoint vs. the converted PyTorch checkpoint.
But since this requires properly running the Jax FNet it requires a bit of setup and some modifications to the official implementation.

#### Verification Setup

1. Clone the official implementation

```bash
svn export https://github.com/google-research/google-research/trunk/f_net
# or
git clone git@github.com:google-research/google-research.git
cd google-research/f_net
```

2. Edit the config in the official implementation to fit the checkpoint you want to run.

3. Add the following to the return value of `_compute_pretraining_metrics` in `models.py`:

```python
return {
    ...
    "masked_lm_logits": masked_lm_logits,
    "next_sentence_logits": next_sentence_logits
}
```

2. Create a `setup.py` file in the parent directory of the `f_net` directory with the following content

```python
from setuptools import setup

setup(
    name='fnet_jax',
    version='0.1.0',
    install_requires=[],
    packages=['f_net']
)
```

3. Install as a dependency in your `fnet-pytorch` project

```bash
pip install -e path/to/dir-of-"setup.py"
```

#### Run the verification script

```bash
python verify_conversion.py \
    --jax path/to/jax_checkpoint \
    --torch path/to/fnet_for_pretraining.statedict.pt \
    --config path/to/config.json \
    --vocab path/to/vocab
```

This should initialize both models from the checkpoints and run inference on a sample text and compare the output logits.