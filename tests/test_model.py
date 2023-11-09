import unittest

from torch import nn, randn

from avae.model import AffinityVAE as avae


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vae = avae(
            capacity=8,
            depth=4,
            input_size=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
        )

        self.vae_2d = avae(
            capacity=8,
            depth=4,
            input_size=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )

    def test_model_instance(self):
        """Test instantiation of the model."""

        assert isinstance(self.vae, avae)

    def test_model_3D(self):
        """Test that model is instantiated with 3D convolutions."""
        assert isinstance(self.vae.encoder.conv_enc[0], nn.Conv3d)
        assert isinstance(self.vae.decoder.conv_dec[-1], nn.ConvTranspose3d)

    def test_model_2D(self):
        """Test that model is instantiated with 2D convolutions."""

        assert isinstance(self.vae_2d.encoder.conv_enc[0], nn.Conv2d)
        assert isinstance(self.vae_2d.decoder.conv_dec[-1], nn.ConvTranspose2d)

    def test_model_eval(self):

        x = randn(14, 1, 64, 64, 64)
        y = self.vae(x)

        self.assertEqual(x.shape, y[0].shape)
        self.assertEqual(randn(14, 16).shape, y[1].shape)
        self.assertEqual(randn(14, 16).shape, y[2].shape)
        self.assertEqual(randn(14, 16).shape, y[3].shape)
        self.assertEqual(randn(14, 3).shape, y[4].shape)
