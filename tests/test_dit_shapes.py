import torch

from src.models.dit import DiT4D


def test_dit4d_forward_shape():
    model = DiT4D(
        input_size=(2, 8, 8, 4),
        patch_size=(1, 4, 4, 2),
        in_channels=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=0,
        learn_sigma=False,
        flash_attention=False,
    )
    x = torch.randn(3, 2, 2, 8, 8, 4)
    t = torch.randint(0, 300, (3,))
    y = model(x, t)
    assert y.shape == x.shape
