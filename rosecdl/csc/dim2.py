import torch
import torch.nn.functional as F
from torch import fft

from rosecdl.csc.base import ConvolutionalSparseCoder


class CSC2d(ConvolutionalSparseCoder):
    def __init__(
        self,
        lmbd,
        n_components=None,
        kernel_size=None,
        n_channels=None,
        D_init=None,
        window=True,
        positive_D=False,
        positive_z=True,
        n_iterations=30,
        deepcdl=False,
        random_state=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            lmbd=lmbd,
            n_components=n_components,
            kernel_size=kernel_size,
            n_channels=n_channels,
            D_init=D_init,
            window=window,
            positive_D=positive_D,
            positive_z=positive_z,
            n_iterations=n_iterations,
            deepcdl=deepcdl,
            random_state=random_state,
            device=device,
            dtype=dtype,
        )
        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

    def normalize_atoms(self) -> None:
        """Renormalize the atoms of the dictionary."""
        with torch.no_grad():
            if self.positive_D:
                # Work on data as _D_hat is a nn.Parameter
                self._D_hat.data = F.relu(self._D_hat.data)
            norm_atoms = torch.linalg.vector_norm(
                self._D_hat, ord=2, dim=(1, 2, 3), keepdim=True
            )
            norm_atoms[torch.nonzero((norm_atoms == 0), as_tuple=False)] = 1
            self._D_hat /= norm_atoms

    def compute_lipschitz(self):
        """Compute the Lipschitz constant using the FFT."""
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
