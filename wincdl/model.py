import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from alphacsc.init_dict import init_dictionary
from alphacsc.utils.dictionary import get_uv, tukey_window


class CSC1d(nn.Module):
    def __init__(
        self,
        n_iterations,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        rank="full",
        window=False,
        D_init=None,
        positive_z=True,
        random_state=None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.n_components = n_components
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.do_window = window

        self.lmbd = lmbd
        self.positive_z = positive_z

        self.n_iterations = n_iterations

        self.dtype = dtype
        self.device = device
        self.random_state = random_state

        self.generator = torch.Generator(device)
        self.generator.manual_seed(random_state)


        # Tukey window
        if window:
            self.window = torch.tensor(
                tukey_window(*self.kernel_size), dtype=dtype, device=device
            )[None, None, :]
        else:
            self.window = None

        # Convolution
        self.conv = F.conv1d
        self.convt = F.conv_transpose1d

        # FISTA operators
        self.prox = (
            lambda x, lmbd: F.relu(x - lmbd) if self.positive_z
            else lambda x, lmbd: x - F.clip(x, -lmbd, lmbd)
        )
        self.grad_loss = lambda x, z, D: self.conv((self.convt(z, D) - x), D)

        # Rank
        self.rank = rank

        self.init_D(D_init)

        # Collect usage statistics for the dictionary elements
        self.reset_usage_statistics()
        self._resampled_atoms = []

        self.to(device=device, dtype=dtype)

    def init_D(self, D_init):

        # Initialisation
        if D_init is None or (isinstance(D_init, str) and D_init == "random"):
            D_hat = torch.randn(
                (self.n_components, self.n_channels, *self.kernel_size),
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            D_hat = torch.tensor(D_init, dtype=self.dtype, device=self.device)

        if self.rank == "uv_constraint":
            u = D_hat[:, : self.n_channels][:, :, None]
            v = D_hat[:, self.n_channels :][:, None, :]

            self.u = nn.Parameter(
                torch.tensor(u, dtype=self.dtype, device=self.device)
            )

            self.v = nn.Parameter(
                torch.tensor(v, dtype=self.dtype, device=self.device)
            )

        elif self.rank == "full":
            self._D_hat = nn.Parameter(
                torch.tensor(
                    D_hat.clone().detach(), dtype=self.dtype, device=self.device
                )
            )

        self.rescale()

    def reset_usage_statistics(self):
        self._z_usage = torch.zeros(
            self.n_components, dtype=torch.int32, device=self.device
        )
        self._processed_samples = 0

    def get_D(self):
        if self.rank == "uv_constraint":
            D = self.u * self.v
        elif self.rank == "full":
            D = self._D_hat
        if self.do_window:
            return D * self.window
        else:
            return D

    @property
    def D_hat_(self):
        return self.get_D().to("cpu").detach().numpy()

    @property
    def uv_hat_(self):
        return get_uv(self.D_hat_)

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            if self.rank == "uv_constraint":
                if self.do_window:
                    norm_col_v = torch.linalg.vector_norm(self.window * self.v, dim=2, keepdim=True)
                else:
                    norm_col_v = torch.linalg.vector_norm(self.v, dim=2, keepdim=True)
                norm_col_v[torch.nonzero((norm_col_v == 0), as_tuple=False)] = 1
                self.v /= norm_col_v

                norm_col_u = torch.linalg.vector_norm(self.u, dim=1, keepdim=True)
                norm_col_u[torch.nonzero((norm_col_u == 0), as_tuple=False)] = 1
                self.u /= norm_col_u
                return norm_col_v, norm_col_u

            elif self.rank == "full":
                if self.do_window:
                    norm_atoms = torch.linalg.vector_norm(
                        self.window * self._D_hat, dim=(1, 2), keepdim=True
                    )
                else:
                    norm_atoms = torch.linalg.vector_norm(self._D_hat, dim=(1, 2), keepdim=True)
                norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1
                self._D_hat /= norm_atoms
                return norm_atoms

    def _resample_atom(self, k0):
        """Resample an atom if it is not used enough """

        # XXX: better resample?
        D_temp = init_dictionary(
            # Only using the shape of X to generate the dictionary
            torch.zeros((self.n_components, self.n_channels, *self.kernel_size)),
            n_atoms=1,
            n_times_atom=self.kernel_size[0],
            rank1=False,
            window=self.do_window,
            D_init="random",
            random_state=self.random_state,
        )
        self._D_hat[k0] = torch.tensor(D_temp, dtype=self.dtype, device=self.device)
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
                    len(null_atom_indices), (1,), generator=self.generator,
                    device=self.device
                )
                k0 = null_atom_indices[idx].item()
                # no resampling of the last 2 resampled atoms
                if k0 in self._resampled_atoms[-2:]:
                    return
                # Create a new random atom and rescale
                self._resample_atom(k0)
                self._resampled_atoms.append(k0)
                print(f"Resampled atom {k0}")

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT
        """
        with torch.no_grad():
            fourier_dico = fft.fft(self.get_D(), dim=2)
            lipschitz = (
                torch.max(torch.real(fourier_dico * torch.conj(fourier_dico)), dim=2)[0]
                .sum()
                .item()
            )
            if lipschitz == 0:
                lipschitz = 1
            return lipschitz

    def init_z(self, x):
        support_shape = tuple(fs - ks + 1 for fs, ks in zip(x.shape[2:], self.kernel_size))
        return torch.zeros(
            (x.shape[0], self.n_components, *support_shape),
            dtype=torch.float, device=self.device,
        )

    def forward(self, x, D=None):
        """
        (F)ISTA-like forward pass

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

        # We don't use unrolling here but alternate minimization
        # TODO: evaluate this choice
        with torch.no_grad():
            # Initialization equal 0
            z_hat = self.init_z(x)
            L = self.compute_lipschitz()
            z_hat = self.fista(z_hat, x, D, self.lmbd, L, n_iter=self.n_iterations)

            # update usage statistics
            # TODO: maybe move to a dedicated function
            self._z_usage += torch.sum(z_hat != 0, dim=tuple(range(2, z_hat.ndim))).sum(dim=0)
            self._processed_samples += z_hat.shape[0] * np.prod(z_hat.shape[2:])

        return self.convt(z_hat, D), z_hat


    def fista(self, zO, x, D, lmbd, L, n_iter):
        """

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


class CSC2d(CSC1d):
    def __init__(
        self,
        n_iterations,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        rank="full",
        window=False,
        D_init=None,
        positive_z=True,
        random_state=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            n_iterations=n_iterations,
            n_components=n_components,
            kernel_size=kernel_size,
            n_channels=n_channels,
            lmbd=lmbd,
            device=device,
            dtype=dtype,
            random_state=2147483647,
            rank="full",
            window=False,
            D_init=None,
            positive_z=True,
        )

        # Convolution
        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            norm_atoms = torch.linalg.vector_norm(self._D_hat, ord=2, dim=(1, 2, 3), keepdim=True)
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1
            self._D_hat /= norm_atoms

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT
        """
        with torch.no_grad():
            fourier_dico = fft.fftn(self._D_hat, dim=(1, 2, 3))
            lipschitz = (
                torch.amax(
                    torch.real(fourier_dico * torch.conj(fourier_dico)), dim=(1, 2, 3)
                )
                .sum()
                .item()
            )
            if lipschitz == 0:
                lipschitz = 1
            return lipschitz

    def _resample_atom(self, k0):
        """Resample an atom if it is not used enough """

        # XXX: better resample?
        D_temp = torch.rand(
            (1, self.n_channels, *self.kernel_size),
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )
        self._D_hat[k0] = D_temp
        self.rescale()
