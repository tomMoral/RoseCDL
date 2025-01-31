import torch

from rosecdl.csc.dim1 import CSC1d


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
