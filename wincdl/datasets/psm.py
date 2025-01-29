import pandas as pd
import requests
from pathlib import Path

URL_XTRAIN = (
    "https://drive.google.com/uc?&id=1d3tAbYTj0CZLhB7z3IDTfTRg3E7qj_tw"
    "&export=download"
)
URL_XTEST = (
    "https://drive.google.com/uc?&id=1RQH7igHhm_0GAgXyVpkJk6TenDl9rd53"
    "&export=download"
)
URL_YTEST = (
    "https://drive.google.com/uc?&id=1SYgcRt0DH--byFbvkKTkezJKU5ZENZhw"
    "&export=download"
)

# Heavily inspired from Benchopt library.
class PSM:
    name = "PSM"

    def __init__(self, debug=False):
        self.debug = debug

    def get_data(self):
        path = Path("data") / self.name

        # Check if the data is already here
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

            response = requests.get(URL_XTRAIN)
            with open(path / "PSM_train.csv", "wb") as f:
                f.write(response.content)
            response = requests.get(URL_XTEST)
            with open(path / "PSM_test.csv", "wb") as f:
                f.write(response.content)
            response = requests.get(URL_YTEST)
            with open(path / "PSM_test_label.csv", "wb") as f:
                f.write(response.content)

        X_train = pd.read_csv(path / "PSM_train.csv")
        X_train.fillna(X_train.mean(), inplace=True)
        X_train = X_train.to_numpy()

        X_test = pd.read_csv(path / "PSM_test.csv")
        X_test.fillna(X_train.mean(), inplace=True)
        X_test = X_test.to_numpy()

        y_test = pd.read_csv(path / "PSM_test_label.csv").to_numpy()[:, 1]

        # Limiting the size of the dataset for testing purposes
        if self.debug:
            X_train = X_train[:1000]
            X_test = X_test[:1000]
            y_test = y_test[:1000]

        # Reshaping the data to have (n_trials, n_channels, n_times)
        X_train = X_train.T[None, ...]
        X_test = X_test.T[None, ...]
        y_test = y_test.astype(int)

        return dict(X_train=X_train, y_test=y_test, X_test=X_test)
