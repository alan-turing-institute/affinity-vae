import unittest

import pytest
from torch import nn, randn

from avae.decoders.decoders import Decoder
from avae.encoders.encoders import Encoder
from avae.models import AffinityVAE as avae


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:

        self.encoder_3d = Encoder(
            capacity=8,
            depth=4,
            input_size=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
            bnorm=True,
        )

        self.decoder_3d = Decoder(
            capacity=8,
            depth=4,
            input_size=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
        )

        self.vae = avae(self.encoder_3d, self.decoder_3d)

        self.encoder_2d = Encoder(
            capacity=8,
            depth=4,
            input_size=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )
        self.decoder_2d = Decoder(
            capacity=8,
            depth=4,
            input_size=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )

        self.vae_2d = avae(self.encoder_2d, self.decoder_2d)

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

    def test_model_filters(self):
        input = randn(14, 1, 64, 64)

        enc = Encoder(
            filters=[8, 16],
            input_size=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )
        dec = Decoder(
            filters=[16],
            input_size=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )
        model = avae(enc, dec)
        output = model(input)[0]
        self.assertEqual(output.shape, input.shape)

    def test_model_noconv(self):
        input = randn(14, 1, 128)

        enc = Encoder(
            input_size=(128,),
            depth=0,
            latent_dims=16,
            pose_dims=3,
        )
        dec = Decoder(
            input_size=(128,),
            depth=0,
            latent_dims=16,
            pose_dims=3,
        )
        model = avae(enc, dec)
        output = model(input)[0]
        self.assertEqual(output.shape, input.shape)


class PoseTest(unittest.TestCase):
    def setUp(self):
        self.input = randn(14, 1, 64, 64, 64)
        self.params = [(0, 2), (1, 3)]  # 3 tensors for pose, 2 without

    def test_encoder(self):
        for dim, exp in self.params:
            enc = Encoder(
                capacity=8,
                depth=4,
                input_size=(64, 64, 64),
                latent_dims=16,
                pose_dims=dim,
                bnorm=True,
            )
            output = enc(self.input)
            self.assertEqual(len(output), exp)
