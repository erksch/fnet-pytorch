import torch
import numpy as np

from fnet import FourierMMLayer, FourierFFTLayer


def test_fourier_matmul():
    max_seq_length = 3
    hidden_size = 8

    inputs = torch.Tensor([
        [1, 0, 0, 11, 9, 2, 0.4, 2],
        [1, 1, 0, 1, 0, 2, 8, 1],
        [1, 4, 0, 5, 5, 0, -3, 1]
    ])

    expected_output = torch.Tensor(np.fft.fftn(inputs).real)

    mm_layer = FourierMMLayer({
        'hidden_size': hidden_size,
        'max_position_embeddings': max_seq_length
    })
    assert torch.all(mm_layer(inputs).eq(expected_output))

    fft_layer = FourierFFTLayer()
    assert torch.allclose(fft_layer(inputs), expected_output)

