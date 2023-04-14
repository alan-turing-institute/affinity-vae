from avae.model_a import AffinityVAE


def test_model_instance():
    """Test instantiation of the model."""
    vae = AffinityVAE()
    assert isinstance(vae, AffinityVAE)
