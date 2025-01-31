from abc import abstractmethod
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rosecdl.utils.dictionary import tukey_window
from rosecdl.utils.utils import get_torch_generator


class ConvolutionalSparseCoder(nn.Module):
    rank1 = False

    def __init__(
        self,
        lmbd,
        n_components=None,
        kernel_size=None,
        n_channels=None,
        D_init=None,
        window=False,
        positive_D=False,
        positive_z=True,
        n_iterations=30,
        deepcdl=False,
        random_state=None,
        device=None,
        dtype=None,
    ):
        self.n_components = n_components
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.do_window = window

        self.deepcdl = deepcdl
        self.n_iterations = n_iterations

        self.lmbd = lmbd
        self.positive_z = positive_z
        self.positive_D = positive_D

        self.dtype = dtype
        self.device = device

        # Control random number generation
        self.generator = get_torch_generator(random_state, device=device)

        # Tukey window
        if window:
            self.window = torch.tensor(
                tukey_window(*self.kernel_size), dtype=dtype, device=device
            )[None, None]
        else:
            self.window = None

        # FISTA operators
        self.prox = lambda x, lmbd: (
            F.relu(x - lmbd)
            if self.positive_z
            else lambda x, lmbd: x - F.clip(x, -lmbd, lmbd)
        )
        self.grad_loss = lambda x, z, D: self.conv((self.convt(z, D) - x), D)

        self.init_D(D_init)

        # Collect usage statistics for the dictionary elements
        self.reset_usage_statistics()
        self._resampled_atoms = []

        self.to(device=device, dtype=dtype)

    @abstractmethod
    def rescale(self):
        pass

    @abstractmethod
    def compute_lipschitz(self):
        pass

    def init_unnormalized_D(self, D_init):
        if D_init is None or (isinstance(D_init, str) and D_init == "random"):
            return torch.randn(
                (self.n_components, self.n_channels, *self.kernel_size),
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
        return torch.tensor(D_init, dtype=self.dtype, device=self.device)

    def init_D(self, D_init):
        D_hat = self.init_unnormalized_D(D_init)
        self._D_hat = nn.Parameter(
            D_hat.clone().detach().to(dtype=self.dtype, device=self.device)
        )
        self.rescale()

    def reset_usage_statistics(self):
        self._z_usage = torch.zeros(
            self.n_components, dtype=torch.int32, device=self.device
        )
        self._processed_samples = 0

    def get_D(self):
        if self.rank1:
            D = self.u * self.v
        else:
            D = self._D_hat
        if self.do_window:
            return D * self.window
        else:
            return D

    @property
    def D_hat_(self):
        return self.get_D().detach().cpu().numpy()

    def _resample_atom(self, k0):
        """Resample an atom if it is not used enough."""
        # XXX: better resample?
        D_temp = torch.rand(
            (1, self.n_channels, *self.kernel_size),
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        self._D_hat[k0] = D_temp
        self.rescale()

    def resample_atom(self):
        with torch.no_grad():
            # compute sparsity, resample unsed atom if needed
            z_nnz = self._z_usage / self._processed_samples
            self.reset_usage_statistics()
            null_atom_indices = torch.where(z_nnz < 1e-3)[0]
            if len(null_atom_indices) > 0:
                # resample a random atom
                idx = torch.randint(
                    len(null_atom_indices),
                    (1,),
                    generator=self.generator,
                    device=self.device,
                )
                k0 = null_atom_indices[idx].item()
                # no resampling of the last 2 resampled atoms
                if k0 in self._resampled_atoms[-2:]:
                    return
                # Create a new random atom and rescale
                self._resample_atom(k0)
                self._resampled_atoms.append(k0)
                print(f"Resampled atom {k0}")

    def init_z(self, x):
        support_shape = tuple(
            fs - ks + 1 for fs, ks in zip(x.shape[2:], self.kernel_size)
        )
        return torch.zeros(
            (x.shape[0], self.n_components, *support_shape),
            dtype=torch.float,
            device=self.device,
        )

    def forward(self, x, D=None):
        """(F)ISTA-like forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (number of samples, channels, time)
            Data to be processed by (F)ISTA

        Returns
        -------
        out : torch.Tensor, shape
            (number of data, n_components,
            time - kernel_size + 1)
            Approximation of the sparse code associated to y
        """
        # Compute current dictionary
        if D is None:
            D = self.get_D()

        # Here, only use unrolling if self.deepcdl, else use alternate minimization
        ctx = nullcontext() if self.deepcdl else torch.no_grad()
        with ctx:
            # Initialization equal 0
            z_hat = self.init_z(x)
            L = self.compute_lipschitz()
            z_hat = self.fista(z_hat, x, D, self.lmbd, L, n_iter=self.n_iterations)

            # update usage statistics
            # TODO: maybe move to a dedicated function
            self._z_usage += torch.sum(z_hat != 0, dim=tuple(range(2, z_hat.ndim))).sum(
                dim=0
            )
            self._processed_samples += z_hat.shape[0] * np.prod(z_hat.shape[2:])

        return self.convt(z_hat, D), z_hat

    def fista(self, zO, x, D, lmbd, L, n_iter):
        """FISTA algorithm.

        Parameters
        ----------
        zO : torch.Tensor, shape (number of samples, n_components, time)
            Initialisation of the sparse code
        x : torch.Tensor, shape (number of samples, channels, time)
            Data to be processed by (F)ISTA
        D : torch.Tensor, shape (n_components, channels, kernel_size)
            Dictionary
        lmbd : float
            Regularization parameter
        L : float
            Lipschitz constant for the data-fit term
        n_iter : int
            Number of iterations
        """
        z = zO
        w = z.clone()
        beta = 1
        for i in range(n_iter):
            w_new = self.prox(z - self.grad_loss(x, z, D) / L, lmbd / L)
            beta_new = (1 + np.sqrt(1 + 4 * beta**2)) / 2
            z_new = w_new + (beta - 1) / beta_new * (w_new - w)
            z, w, beta = z_new, w_new, beta_new

        return w_new
