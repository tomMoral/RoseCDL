import torch

from rosecdl.utils.utils_outliers import get_outlier_mask


def test_get_outlier_mask_1d():
    start_outlier = 5
    len_sig, len_outlier = 100, 2

    outliers = slice(start_outlier, start_outlier + len_outlier + 1)
    err = torch.zeros(1, 1, len_sig)
    err[0, 0, outliers] = 1

    mask = get_outlier_mask(err, threshold=2 * len_outlier / len_sig)
    assert torch.all(mask == err)

    opening_window = 3
    mask = get_outlier_mask(err, threshold=0.8, opening_window=opening_window)
    expected_mask = torch.zeros(1, 1, len_sig)
    masked_times = slice(
        max(0, start_outlier - (opening_window // 2)),
        start_outlier + len_outlier + (opening_window) // 2 + 1,
    )
    expected_mask[0, 0, masked_times] = 1
    assert torch.all(mask == expected_mask)
