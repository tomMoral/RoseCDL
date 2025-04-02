import torch
import torch.nn.functional as f
from torch import fft, nn

from rosecdl.csc.base import ConvolutionalSparseCoder
from rosecdl.utils.convolution import fft_conv, fft_conv_transpose
from rosecdl.utils.dictionary import get_uv, tukey_window


class CSC1d(ConvolutionalSparseCoder):
    """Convolutional Sparse (en)Coder for 1D signals."""

    def set_conv_methods(self) -> None:
        if self.conv_algo == "classical":
            self.conv = f.conv1d
            self.convt = f.conv_transpose1d
        elif self.conv_algo == "fft":
            self.conv = fft_conv
            self.convt = fft_conv_transpose
        else:
            raise ValueError(f"Unknown convolution algorithm: {self.conv_algo}")

    def tukey_window(self) -> torch.Tensor:
        """N-dimensional Tukey window."""
        return torch.tensor(
            tukey_window(*self.kernel_size), dtype=self.dtype, device=self.device
        )[None, None]

    def normalize_atoms(self) -> None:
        """Normalize the atoms of the dictionary."""
        with torch.no_grad():
            if self.positive_D:
                # Work on data as _D_hat is a nn.Parameter
                self._D_hat.data = f.relu(self._D_hat.data)

            if self.do_window:
                norm_atoms = torch.linalg.vector_norm(
                    self.window * self._D_hat, dim=(1, 2), keepdim=True
                )
            else:
                norm_atoms = torch.linalg.vector_norm(
                    self._D_hat, dim=(1, 2), keepdim=True
                )
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1
            self._D_hat /= norm_atoms

    def compute_lipschitz(self) -> float:
        r"""Compute the Lipschitz constant of the gradient w.r.t. z.

        This gradient writes as:
        \begin{equation}
            \nabla_{z} F = D^{\Lsh} * (D * z - X),
        \end{equation}
        where $D^{\Lsh}$ (a.k.a. "$D$ reverse") is obtained by reverting
        the values of $D$ along all its axes. See more in Cédric's PhD thesis,
        Appendix B.
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


class Rank1CSC1d(CSC1d):
    """1D Convolutional Sparse (en)Coder with rank-1 constraint."""

    @property
    def uv_hat_(self):
        return get_uv(self.D_hat_)

    def normalize_atoms(self) -> None:
        """Normalize the atoms of the dictionary."""
        with torch.no_grad():
            if self.positive_D:
                # Work on data as u, v are nn.Parameter
                self.u.data = f.relu(self.u.data)
                self.v.data = f.relu(self.v.data)

            if self.do_window:
                norm_col_v = torch.linalg.vector_norm(
                    self.window * self.v, dim=2, keepdim=True
                )
            else:
                norm_col_v = torch.linalg.vector_norm(self.v, dim=2, keepdim=True)
            norm_col_v[torch.nonzero((norm_col_v == 0), as_tuple=False)] = 1
            self.v /= norm_col_v

            norm_col_u = torch.linalg.vector_norm(self.u, dim=1, keepdim=True)
            norm_col_u[torch.nonzero((norm_col_u == 0), as_tuple=False)] = 1
            self.u /= norm_col_u

    def get_D(self):
        D = self.u * self.v
        if self.do_window:
            return D * self.window
        return D

    def init_unnormalized_D(self, D_init):
        if D_init is None or (isinstance(D_init, str) and D_init == "random"):
            return torch.randn(
                (self.n_components, self.n_channels + self.kernel_size[0]),
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
        return D_init.clone().detach().to(self.dtype).to(self.device)

    def init_D(self, D_init):
        """Initialize dictionary and normalize its atoms.

        In this cas we have uv constraint so u and v are actually initialized.
        WARNING: Note that the shape differs from the non-rank1 case!!!

        Args:
            D_init: tensor of shape (n_atoms, n_channels + n_times_atom)

        """
        D_hat = self.init_unnormalized_D(D_init)

        n_channels = self.n_channels  # Just to avoid Flake8 flag in the slices below
        u = torch.unsqueeze(D_hat[:, :n_channels], dim=2)
        v = torch.unsqueeze(D_hat[:, n_channels:], dim=1)

        self.u = nn.Parameter(u.clone().detach().to(self.dtype).to(self.device))
        self.v = nn.Parameter(v.clone().detach().to(self.dtype).to(self.device))
        self.normalize_atoms()

        self.n_components, self.n_channels, *self.kernel_size = self.D_hat_.shape
        self.n_components, self.n_channels, *self.kernel_size = self.D_hat_.shape
