# FNet PyTorch

## Pre-Training

You can pre-train an FNet from a checkpoint or from scratch.

Keep in mind that can also always use the official implementation for training and converting the resulting checkpoint.

### Setup

1) Create a virtualenv and install dependencies

```bash
pip install -r training/requirements.txt
```

2) Copy the `example.ini` and configure it to your needs.


### Start a pre-training

Run a training (from this repositories root directory)

```bash
python -m training.pretraining --config myconfig.ini
```
