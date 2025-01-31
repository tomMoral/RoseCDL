import torch
import torch.nn.functional as F
from torch import fft

from rosecdl.csc.base import ConvolutionalSparseCoder
from rosecdl.utils.dictionary import get_uv


class CSC1d(ConvolutionalSparseCoder):
    def __init__(
        self,
        lmbd,
        n_components=None,
        kernel_size=None,
        n_channels=None,
        D_init=None,
        rank1=False,
        window=False,
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
            rank1=rank1,
            window=window,
            positive_D=positive_D,
            positive_z=positive_z,
            n_iterations=n_iterations,
            deepcdl=deepcdl,
            random_state=random_state,
            device=device,
            dtype=dtype,
        )
        self.conv = F.conv1d
        self.convt = F.conv_transpose1d

    @property
    def uv_hat_(self):
        return get_uv(self.D_hat_)

    def rescale(self):
        """Constrains the dictionary to have normalized atoms."""
        with torch.no_grad():
            if self.rank1:
                if self.positive_D:
                    # Work on data as u, v are nn.Parameter
                    self.u.data = F.relu(self.u.data)
                    self.v.data = F.relu(self.v.data)

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
                return norm_col_v, norm_col_u

            else:
                if self.positive_D:
                    # Work on data as _D_hat is a nn.Parameter
                    self._D_hat.data = F.relu(self._D_hat.data)

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
                return norm_atoms

    def compute_lipschitz(self):
        """Compute the Lipschitz constant using the FFT."""
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
