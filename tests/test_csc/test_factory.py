from rosecdl.csc.factory import csc_factory


def test_csc_factory():

    base_dict = {
        "lmbd": 0.1,
        "n_components": 4,
        "n_channels": 10,
        "D_init": None,
        "window": False,
        "positive_D": True,
        "positive_z": True,
        "n_iterations": 20,
        "deepcdl": False,
        "random_state": None,
        "device": None,
        "dtype": float,
    }

    io_pair_list = [
        {"input": {"kernel_size": (10,), "rank1": False}, "output": "CSC1d"},
        {"input": {"kernel_size": (10,), "rank1": True}, "output": "CSC1d"},
        {"input": {"kernel_size": (10, 10), "rank1": False}, "output": "CSC1d"},
    ]

    for io_pair in io_pair_list:
        kwargs = {**base_dict, **io_pair["input"]}
        assert type(csc_factory(**kwargs)).__name__ == io_pair["output"]
