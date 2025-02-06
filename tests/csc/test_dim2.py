import numpy as np
import torch

from rosecdl.csc.dim2 import CSC2d


class TestCSC2d:

    @staticmethod
    def get_base_config():
        return {
            "lmbd": 0.1,
            "n_components": 4,
            "n_channels": 2,
            "D_init": None,
            "window": False,
            "positive_D": True,
            "positive_z": True,
            "n_iterations": 20,
            "deepcdl": False,
            "random_state": None,
            "device": torch.device("cpu"),
            "dtype": float,
        }

    def test_init(self):
        base_config = self.get_base_config()
        csc = CSC2d(**base_config, kernel_size=(10, 10))
        for k, v in base_config.items():
            if k == "D_init":
                assert csc._D_hat.shape == (
                    base_config["n_components"],
                    base_config["n_channels"],
                    10,
                    10,
                )
                continue
            if k == "window":
                assert csc.do_window == v
                assert csc.window is None
                continue
            if k == "random_state":
                assert type(csc.generator.seed()) is int
                continue
            assert getattr(csc, k) == v

    def test_init_with_dict(self):
        config = self.get_base_config()
        for k in ["n_components", "n_channels", "kernel_size"]:
            config[k] = None
        config["D_init"] = torch.zeros(4, 2, 10, 10)
        csc = CSC2d(**config)
        assert csc.n_components == 4
        assert csc.n_channels == 2
        assert csc.kernel_size == (10, 10)

    def test_normalize_atoms(self):
        base_config = self.get_base_config()

        n_components = base_config["n_components"]
        n_channels = base_config["n_channels"]
        kernel_size = (10, 10)

        base_config["D_init"] = torch.ones((n_components, n_channels, *kernel_size))
        csc = CSC2d(**base_config, kernel_size=kernel_size)

        epsilon = 1e-10

        csc.state_dict()["_D_hat"] *= 2
        assert (
            torch.linalg.vector_norm(csc._D_hat, dim=(1, 2, 3), keepdim=True).max()
            > 1 + epsilon
        )
        csc.normalize_atoms()
        assert (
            torch.linalg.vector_norm(csc._D_hat, dim=(1, 2, 3), keepdim=True).max()
            <= 1 + epsilon
        )

    def test_compute_lipschitz(self):
        base_config = self.get_base_config()

        n_components = base_config["n_components"]
        n_channels = base_config["n_channels"]
        kernel_size = (10, 10)

        d_init = torch.zeros((n_components, n_channels, *kernel_size))
        d_init[:, :, 0, 0] = 1
        base_config["D_init"] = d_init

        csc = CSC2d(**base_config, kernel_size=kernel_size)

        assert np.allclose(csc.compute_lipschitz(), n_channels * n_components)
