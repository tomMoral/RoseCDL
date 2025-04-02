import torch
import torch.nn.functional as f
from torch import fft

from rosecdl.csc.base import ConvolutionalSparseCoder
from rosecdl.utils.convolution import fft_conv, fft_conv_transpose
from rosecdl.utils.dictionary import tukey_window_2d


class CSC2d(ConvolutionalSparseCoder):
    """Convolutional Sparse (en)Coder for 2D signals."""

    def set_conv_methods(self) -> None:
        if self.conv_algo == "classical":
            self.conv = f.conv2d
            self.convt = f.conv_transpose2d
        elif self.conv_algo == "fft":
            self.conv = fft_conv
            self.convt = fft_conv_transpose
        else:
            raise ValueError(f"Unknown convolution algorithm: {self.conv_algo}")

    def tukey_window(self) -> torch.Tensor:
        """N-dimensional Tukey window."""
        return torch.tensor(
            tukey_window_2d(*self.kernel_size), dtype=self.dtype, device=self.device
        )[None, None]

    def normalize_atoms(self) -> None:
        """Normalize the atoms of the dictionary."""
        with torch.no_grad():
            if self.positive_D:
                # Work on data as _D_hat is a nn.Parameter
                self._D_hat.data = f.relu(self._D_hat.data)
            norm_atoms = torch.linalg.vector_norm(
                self._D_hat, ord=2, dim=(1, 2, 3), keepdim=True
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
            return lipschitz
