from torch import nn

from avae.model_a import AffinityVAE as avae_a
from avae.model_b import AffinityVAE as avae_b


def test_model_instance_a():
    """Test instantiation of the model a."""
    vae = avae_a()
    assert isinstance(vae, avae_a)


def test_model_instance_b():
    """Test instantiation of the model b."""
    vae = avae_b()
    assert isinstance(vae, avae_b)


def test_model_3D():
    """Test that model is instantiated with 3D convolutions."""
    vae = avae_a(
        capacity=8,
        depth=4,
        input_shape=(1, 64, 64, 64),
        latent_dims=16,
        pose_dims=3,
    )
    assert isinstance(vae.encoder[0], nn.Conv3d)
    assert isinstance(vae.decoder[-1], nn.ConvTranspose3d)


def test_model_2D():
    """Test that model is instantiated with 2D convolutions."""
    vae = avae_a(
        capacity=8,
        depth=4,
        input_shape=(1, 64, 64),
        latent_dims=16,
        pose_dims=3,
    )
    assert isinstance(vae.encoder[0], nn.Conv2d)
    assert isinstance(vae.decoder[-1], nn.ConvTranspose2d)
