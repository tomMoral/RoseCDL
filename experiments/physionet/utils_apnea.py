import numpy as np
from pathlib import Path
from scipy.signal import tukey

import wfdb
from wfdb.io.record import rdrecord
from wfdb.io.annotation import rdann




def load_ecg(subject="a01", T=60, data_path=Path('apnea-ecg'),
             apply_window=True, verbose=True):    
    """

    Parameters
    ----------
    subject : str
        subject name

    T : float
        duration, in seconds, of data splits
        default is 60 as the data have been annoted by the minute

    data_path : pathlib.Path
        path to data folder

    apply_window : bool (default: True)
        If set to True (default), a tukey window is applied to each split to
        reduce the border artifacts by reducing the weights of the chunk
        borders.

    verbose : bool
        if True, will print some information


    Returns
    -------
    data : ndarray, shape (n_splits, n_channels, int(T * fs))
        The signal splitted in ``n_splits``,
        whith ``n_splits = sig_len // int(T * fs)``,
        fs being the sampling frequency of the record

    labels : 1d array
        labels corresponding to one minute segments
        i.e., if T = 60, labels have the same length as data and each label
        corresponds to each datta split.
    
    """

    # ECG record
    record_name = str(data_path / subject)
    ecg_record = rdrecord(record_name=record_name)

    # split signal
    fs = ecg_record.fs  # sampling frequency of the record
    if verbose:
        print(f"Sampling frequency of the record: {fs} Hz")
    n_times = int(T * fs)
    n_splits = ecg_record.sig_len // n_times
    X = ecg_record.p_signal[:n_splits*n_times, :].T
    X = X.reshape(ecg_record.n_sig, n_splits, n_times).swapaxes(0, 1)

    # Apply a window to the signal to reduce the border artifacts
    if apply_window:
        X *= tukey(n_times, alpha=0.1)[None, None, :]

    # Add labels 
    ann = rdann(
        record_name=record_name,
        extension='apn',
        return_label_elements=['symbol'],
        summarize_labels=True,
    )
    labels = np.array(ann.symbol)
    if T == 60:
        # ensure that labels and data have the same number of trials
        n_trials = min(len(ann.symbol), n_splits)
        return X[:n_trials], labels[:n_trials]
    else:
        import warnings
        warnings.warn('The returned labels do not match the data as T != 60 '
                      f'(got T = {T}).')
        return X, labels