from torch import nn, randn, Size

from avae.model_a import AffinityVAE as avae_a
from avae.model_b import AffinityVAE as avae_b
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vae_a = avae_a(
            capacity=8,
            depth=4,
            input_size=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
        )
        self.vae_b = avae_b(
            capacity=8,
            depth=4,
            input_size=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
        )
        self.vae_a_2d = avae_a(
            capacity=8,
            depth=4,
            input_size=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )


    def test_model_instance_a(self):
        """Test instantiation of the model a."""

        assert isinstance(self.vae_a, avae_a)


    def test_model_instance_b(self):
        """Test instantiation of the model a."""

        assert isinstance(self.vae_b, avae_b)


    def test_model_a_3D(self):
        """Test that model is instantiated with 3D convolutions."""
        assert isinstance(self.vae_a.encoder.encoder[0], nn.Conv3d)
        assert isinstance(self.vae_a.decoder.decoder[-1], nn.ConvTranspose3d)


    def test_model_a_2D(self):
        """Test that model is instantiated with 2D convolutions."""

        assert isinstance(self.vae_a_2d.encoder.encoder[0], nn.Conv2d)
        assert isinstance(self.vae_a_2d.decoder.decoder[-1], nn.ConvTranspose2d)

    assert isinstance(vae.encoder.encoder[0], nn.Conv2d)
    assert isinstance(vae.decoder.decoder[-1], nn.ConvTranspose2d)
