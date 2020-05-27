from setuptools import setup

setup(name='mollvae',
      version='1.0',
      # list folders, not files
      packages=["mollvae",
                "mollvae.model",
                "mollvae.utils",
                "mollvae.tests",
                "mollvae.model.encoders",
                "mollvae.model.decoders"]
      )