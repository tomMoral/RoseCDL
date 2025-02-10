import torch
import torch.nn.functional as F
from torch import fft

from rosecdl.csc.base import ConvolutionalSparseCoder
from rosecdl.utils.dictionary import tukey_window_2d


class CSC2d(ConvolutionalSparseCoder):
    """Convolutional Sparse (en)Coder for 2D signals."""

    def __init__(
        self,
        lmbd: float,
        n_components: int | None = None,
        kernel_size: tuple | None = None,
        n_channels: int | None = None,
        D_init: torch.Tensor | None = None,
        window: bool = False,
        positive_D: bool = False,
        positive_z: bool = True,
        n_iterations: int = 30,
        deepcdl: bool = False,
        random_state: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the object.

        Args:
            lmbd: regularization parameter,
            n_components: number of atoms in the dictionary,
            kernel_size: support of the atoms (tuple of lenght n for n-dim signals),
            n_channels: number of channels in the signal,
            D_init: optional initial dictionary,
            window: whether a tukey window should be applied to the atoms,
            positive_D: whether to impose the atoms to have positive values,
            positive_z: whether to impose the the activation vectors
                to have positive values,
            n_iterations: number of approximate sparse coding iteration,
            deepcdl: True if unrolled sparse coding, else False,
            random_state: seed for dictionary initialization and atom resampling,
            device: device where the parameters are stored,
            dtype: data type of the parameters.

        """
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
                self._D_hat.data = F.relu(self._D_hat.data)
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
