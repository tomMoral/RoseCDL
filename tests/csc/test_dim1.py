import numpy as np
import torch

from rosecdl.csc.dim1 import CSC1d, Rank1CSC1d


class TestCSC1d:

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
        csc = CSC1d(**base_config, kernel_size=(10,))
        for k, v in base_config.items():
            if k == "D_init":
                assert csc._D_hat.shape == (
                    base_config["n_components"],
                    base_config["n_channels"],
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

    def test_normalize_atoms(self):
        base_config = self.get_base_config()

        n_components = base_config["n_components"]
        n_channels = base_config["n_channels"]
        atom_length = 10

        base_config["D_init"] = torch.ones((n_components, n_channels, atom_length))
        csc = CSC1d(**base_config, kernel_size=(atom_length,))

        csc.state_dict()["_D_hat"] *= 2
        assert torch.linalg.vector_norm(csc._D_hat, dim=(1, 2), keepdim=True).max() > 1

        csc.normalize_atoms()
        assert torch.linalg.vector_norm(csc._D_hat, dim=(1, 2), keepdim=True).max() <= 1

    def test_compute_lipschitz(self):
        base_config = self.get_base_config()

        n_components = base_config["n_components"]
        n_channels = base_config["n_channels"]
        atom_length = 10

        d_init = torch.zeros((n_components, n_channels, atom_length))
        d_init[:, :, 0] = 1
        base_config["D_init"] = d_init

        csc = CSC1d(**base_config, kernel_size=(atom_length,))

        assert np.allclose(csc.compute_lipschitz(), n_components)


class TestRank1CSC1d(TestCSC1d):

    def test_init(self):
        base_config = self.get_base_config()
        csc = Rank1CSC1d(**base_config, kernel_size=(10,))
        for k, v in base_config.items():
            if k == "D_init":
                assert csc.u.shape == (
                    base_config["n_components"],
                    base_config["n_channels"],
                    1,
                )
                assert csc.v.shape == (base_config["n_components"], 1, 10)
                assert csc.D_hat_.shape == (
                    base_config["n_components"],
                    base_config["n_channels"],
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

    def test_normalize_atoms(self):
        base_config = self.get_base_config()

        n_components = base_config["n_components"]
        n_channels = base_config["n_channels"]
        atom_length = 10

        base_config["D_init"] = torch.ones((n_components, n_channels + atom_length))
        csc = Rank1CSC1d(**base_config, kernel_size=(atom_length,))

        csc.state_dict()["u"] *= 2
        csc.state_dict()["v"] *= 2
        assert torch.linalg.vector_norm(csc.u, dim=(1, 2), keepdim=True).max() > 1
        assert torch.linalg.vector_norm(csc.v, dim=(1, 2), keepdim=True).max() > 1

        csc.normalize_atoms()
        assert torch.linalg.vector_norm(csc.u, dim=(1, 2), keepdim=True).max() <= 1
        assert torch.linalg.vector_norm(csc.v, dim=(1, 2), keepdim=True).max() <= 1
