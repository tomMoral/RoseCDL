import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from alphacsc.utils.dictionary import tukey_window
from alphacsc.update_d_multi import prox_uv

from .utils import get_max_error_patch


class CSC1d(nn.Module):
    def __init__(
        self,
        n_iterations,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        device,
        dtype,
        random_state=2147483647,
        rank="full",
        window=False,
        D_init=None,
        positive_z=True,
    ):
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.lmbd = lmbd
        self.n_channels = n_channels
        self.n_iterations = n_iterations

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.positive_z = positive_z

        # Tukey window
        if window:
            self.window = torch.tensor(
                tukey_window(self.kernel_size), dtype=self.dtype, device=self.device
            )[None, None, :]
        else:
            self.window = None

        # Convolution
        self.conv = F.conv1d
        self.convt = F.conv_transpose1d

        # Rank
        self.rank = rank

        # Initialisation
        if D_init is None:
            D_hat = torch.rand(
                (n_components, n_channels, kernel_size),
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
                torch.tensor(u, dtype=torch.float, device=self.device)
            )

            self.v = nn.Parameter(
                torch.tensor(v, dtype=torch.float, device=self.device)
            )

        elif self.rank == "full":
            # self._D_hat = nn.Parameter(
            #     torch.tensor(
            #         D_hat.clone().detach(), dtype=torch.float, device=self.device
            #     )
            # )
            self._D_hat = nn.Parameter(D_hat.clone().detach().requires_grad_(True))

        self.rescale()

    def get_D(self):
        if self.rank == "uv_constraint":
            D = self.u * self.v
        elif self.rank == "full":
            D = self._D_hat
        if self.window is not None:
            return D * self.window
        else:
            return D

    @property
    def D_hat_(self):
        return self.get_D().to("cpu").detach().numpy()

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            if self.rank == "uv_constraint":
                if self.window is not None:
                    norm_col_v = torch.norm(self.window * self.v, dim=2, keepdim=True)
                else:
                    norm_col_v = torch.norm(self.v, dim=2, keepdim=True)
                norm_col_v[torch.nonzero((norm_col_v == 0), as_tuple=False)] = 1
                self.v /= norm_col_v

                norm_col_u = torch.norm(self.u, dim=1, keepdim=True)
                norm_col_u[torch.nonzero((norm_col_u == 0), as_tuple=False)] = 1
                self.u /= norm_col_u
                return norm_col_v, norm_col_u

            elif self.rank == "full":
                if self.window is not None:
                    norm_atoms = torch.norm(
                        self.window * self._D_hat, dim=(1, 2), keepdim=True
                    )
                else:
                    norm_atoms = torch.norm(self._D_hat, dim=(1, 2), keepdim=True)
                norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1
                self._D_hat /= norm_atoms
                return norm_atoms

    def get_max_error_dict(self, X):
        d0 = get_max_error_patch(X, self.z.cpu().numpy(), self._D_hat.cpu().numpy())
        d0 = torch.tensor(d0, dtype=torch.float, device=self.device)
        d0 *= self.window

        return prox_uv(
            d0.cpu().numpy(), uv_constraint="separate", n_channels=self.n_channels
        )

    def resample_atom(self, k0, X):
        """ """
        # new_atom = torch.tensor(
        #             self.get_max_error_dict(X)[0],
        #             dtype=torch.float,
        #             device=self.device
        #         )
        # self._D_hat[k0] = new_atom / torch.norm(new_atom)
        from alphacsc.init_dict import init_dictionary

        D_temp = init_dictionary(
            X,
            n_atoms=1,
            n_times_atom=self.kernel_size,
            rank1=False,
            window=True,
            D_init="chunk",
            random_state=None,
        )
        self._D_hat[k0] = torch.tensor(D_temp, dtype=torch.float, device=self.device)
        return self._D_hat

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

    def forward(self, x, n_iter=None):
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
        D = self.get_D()

        if n_iter is None:
            n_iter = self.n_iterations

        with torch.no_grad():

            def fista(wO, lmbd, L, prox, grad_loss, n_iter):
                """

                Parameters
                ----------
                prox : callable

                grad_loss : callable

                """
                w = wO
                z = w.clone()
                beta = 1
                for i in range(n_iter):
                    w_new = prox(z - grad_loss(z) / L, lmbd / L)
                    beta_new = (1 + np.sqrt(1 + 4 * beta**2)) / 2
                    z_new = w_new + (beta - 1) / beta_new * (w_new - w)
                    z, w, beta = z_new, w_new, beta_new

                return w_new

            def prox(x, lmbd):
                """Soft thresholding"""
                return F.relu(x - lmbd)

            def grad_loss(z):
                return self.conv((self.convt(z, D) - x), D)

            # Initialization equal 0
            z = torch.zeros(
                (x.shape[0], self.n_components, x.shape[2] - self.kernel_size + 1),
                dtype=torch.float,
                device=self.device,
            )
            L = self.compute_lipschitz()
            self.z = fista(z, self.lmbd, L, prox, grad_loss, n_iter=n_iter)

        return self.convt(self.z, D)

        # # Initialization equal 0
        # out = torch.zeros(
        #     (x.shape[0],
        #      self.n_components,
        #      x.shape[2] - self.kernel_size + 1),
        #     dtype=torch.float,
        #     device=self.device
        # )

        # out_old = out.clone()
        # t_old = 1

        # # Compute steps with Lipschitz constant
        # step = 1. / self.compute_lipschitz()

        # for i in range(self.n_iterations):
        #     # Gradient descent
        #     result1 = self.convt(out, D)
        #     result2 = self.conv(
        #         (result1 - x),
        #         D
        #     )

        #     out = out - step * result2

        #     if not self.positive_z:
        #         out = out - torch.clip(
        #             out,
        #             - step * self.lmbd,
        #             step * self.lmbd
        #         )
        #     else:
        #         thresh = out - step * self.lmbd
        #         out = F.relu(thresh)

        #     # FISTA
        #     t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
        #     z = out + ((t_old-1) / t) * (out - out_old)
        #     out_old = out.clone()
        #     t_old = t
        #     out = z

        # save z vector as atribute
        # self.z = z


class CSC2d(nn.Module):
    def __init__(
        self,
        n_iterations,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        device,
        dtype,
        random_state=2147483647,
        D_init=None,
        positive_z=True,
    ):
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.lmbd = lmbd
        self.n_channels = n_channels
        self.n_iterations = n_iterations

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.positive_z = positive_z

        # Convolution
        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

        # Initialisation
        if D_init is None:
            self._D_hat = nn.Parameter(
                torch.rand(
                    (n_components, n_channels, kernel_size, kernel_size),
                    generator=self.generator,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self._D_hat = nn.Parameter(
                torch.tensor(D_init, dtype=torch.float, device=self.device)
            )

        self.rescale()

    @property
    def D_hat_(self):
        return self._D_hat.to("cpu").detach().numpy()

    def rescale(self):
        """
        Constrains the dictionary to have normalized atoms
        """
        with torch.no_grad():
            norm_atoms = torch.norm(self._D_hat, dim=(1, 2, 3), keepdim=True, p=2)
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

    def forward(self, x):
        """
        (F)ISTA-like forward pass
        Parameters
        ----------
        x : torch.Tensor, shape
            (number of samples, n_channels, patch_size, patch_size)
            Data to be processed by (F)ISTA
        Returns
        -------
        out : torch.Tensor, shape
            (number of samples, patch_size, patch_size)
            Reconstruction with dictionary
        """
        with torch.no_grad():
            # Initialization equal 0
            out = torch.zeros(
                (
                    x.shape[0],
                    self.n_components,
                    x.shape[2] - self.kernel_size + 1,
                    x.shape[3] - self.kernel_size + 1,
                ),
                dtype=torch.float,
                device=self.device,
            )

            out_old = out.clone()
            t_old = 1

            # Compute steps with Lipschitz constant
            step = 1.0 / self.compute_lipschitz()

            for i in range(self.n_iterations):
                # Gradient descent
                result1 = self.convt(out, self._D_hat)
                result2 = self.conv((result1 - x), self._D_hat)

                out = out - step * result2

                if not self.positive_z:
                    out = out - torch.clip(out, -step * self.lmbd, step * self.lmbd)
                else:
                    thresh = out - step * self.lmbd
                    out = F.relu(thresh)

                # FISTA
                t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
                z = out + ((t_old - 1) / t) * (out - out_old)
                out_old = out.clone()
                t_old = t
                out = z

        return self.convt(out, self._D_hat)
