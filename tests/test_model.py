from torch import nn

from affinivae.model import AffinityVAE


def test_model_instance():
    """Test instantiation of the model."""
    vae = AffinityVAE()
    assert isinstance(vae, AffinityVAE)


def test_model_3D():
    """Test that model is instantiated with 3D convolutions."""
    vae = AffinityVAE(input_shape=(1, 64, 64, 64))
    assert isinstance(vae.encoder[0], nn.Conv3d)
    assert isinstance(vae.decoder[-1], nn.ConvTranspose3d)


def test_model_2D():
    """Test that model is instantiated with 2D convolutions."""
    vae = AffinityVAE(input_shape=(1, 64, 64))
    assert isinstance(vae.encoder[0], nn.Conv2d)
    assert isinstance(vae.decoder[-1], nn.ConvTranspose2d)
