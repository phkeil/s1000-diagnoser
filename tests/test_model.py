import torch

from src.model import EfficientDiagnoser, MotorAutoencoder


def test_motor_autoencoder_forward_preserves_input_shape():
    model = MotorAutoencoder(bottleneck_size=128)
    model.eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape


def test_motor_autoencoder_respects_custom_bottleneck_size():
    model = MotorAutoencoder(bottleneck_size=32)

    assert model.fc_bottleneck.out_features == 32
    assert model.decoder_fc.in_features == 32


def test_motor_autoencoder_single_sample_forward_in_eval_mode():
    # BatchNorm1d raises on a batch of size 1 in train mode; eval mode (the
    # inference path in src/inference.py) must handle it via running stats.
    model = MotorAutoencoder()
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape


def test_efficient_diagnoser_forward_returns_logits_and_features():
    model = EfficientDiagnoser(num_classes=2)
    model.eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits, features = model(x)

    assert logits.shape == (2, 2)
    assert features.ndim == 2
    assert features.shape[0] == 2
