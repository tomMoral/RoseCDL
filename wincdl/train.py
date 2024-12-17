import time

import numpy as np
import torch
from tqdm import tqdm

from .utils import get_z_nnz


def compute_objective(X, X_hat, z_hat, reg):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    loss = loss_fn(X, X_hat) / (2 * X.shape[0])
    loss += reg * (z_hat.sum() / z_hat.shape[0]).item()
    return loss


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    max_batch=None,
    scheduler=None,
    resamp_atom=[],
):
    avg_loss = 0
    count = 0
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        # loss = loss_fn(model(X), X)/2
        # loss = loss_fn(model(X), X) / (2 * X.shape[0])

        # # compute Lasso loss
        # # avg_loss += loss.item() + model.lmbd * model.z.sum(axis=(1, 2)).item()
        # avg_loss += loss.item()
        # avg_loss += model.lmbd * (model.z.sum() / model.z.shape[0]).item()

        loss = compute_objective(X, X_hat=model(X), z_hat=model.z, reg=model.lmbd)
        avg_loss += loss.item()

        count += 1

        def closure():
            with torch.no_grad():
                # return loss_fn(model(X), X)
                return compute_objective(
                    X, X_hat=model(X), z_hat=model.z, reg=model.lmbd
                )

        # Backpropagation
        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()
        model.rescale()

        # if True:
        with torch.no_grad():
            # compute sparsity, resample unsed atom if needed
            z_nnz = get_z_nnz(model.z.to("cpu").detach().numpy())
            null_atom_indices = np.where(z_nnz < 2)[0]
            if len(null_atom_indices) > 0:
                for k0 in null_atom_indices:
                    if (
                        k0 in resamp_atom[-2:]
                    ):  # no resampling of the last 2 resampled atoms
                        continue
                    # k0 = null_atom_indices[0]  # only the first one? why so?
                    model.resample_atom(k0, X.cpu().numpy())
                    print("Resampled atom {}".format(k0))
                    resamp_atom.append(k0)
                    break

        if scheduler is not None:
            scheduler.step()

        if max_batch is not None and batch >= max_batch:
            break

    return avg_loss / count, resamp_atom


def train(
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    scheduler=None,
    epochs=10,
    max_batch=None,
    save_list_D=False,
    stopping_criterion=True,
    tol=1e-8,
):
    """
    Training process

    Parameters
    ----------
    model : torch.nn.Module
        Torch network
    train_dataloader : torch.utils.data.DataLoader
        Train dataset
    optimizer : torch.optim.Optimizer
        Torch optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning step size scheduler, by default None
    epochs : int, optional
        Number of epochs, by default 10
    max_batch : int, optional
        Maximum number of minibatches per epoch, by default None

    Returns
    -------
    list, list
        Train losses, test losses
    """
    old_loss = None
    train_losses = []
    list_D = []
    times = []
    if save_list_D:
        list_D.append(model.D_hat_)
        start = time.time()
        times.append(0)
    # compute init loss
    _, X = list(enumerate(train_dataloader))[0]
    z_init = torch.zeros(
        (X.shape[0], model.n_components, X.shape[2] - model.kernel_size + 1),
        dtype=torch.float,
        device=model.device,
    )
    model.z = z_init
    D_init = model.get_D()
    X_hat = model.convt(model.z, D_init)
    with torch.no_grad():
        # loss_init = loss_fn(X_hat, X).item() / (2 * X.shape[0])
        # # loss_init += model.lmbd * model.z.sum(axis=(1, 2)).item()
        # loss_init += model.lmbd * (model.z.sum() / model.z.shape[0]).item()
        loss_init = compute_objective(X, X_hat, z_hat=model.z, reg=model.lmbd)
        train_losses.append(loss_init.item())

    pbar = tqdm(range(epochs))
    resamp_atom = []

    for epoch in pbar:
        train_loss, resamp_atom = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            max_batch=max_batch,
            scheduler=scheduler,
            resamp_atom=resamp_atom,
        )

        train_losses.append(train_loss)
        if save_list_D:
            list_D.append(model.D_hat_)
            times.append(time.time() - start)

        pbar.set_description(
            f"Epoch {epoch+1}"
            f" - Average train loss: {train_loss:.15f}"
            f" - Step size: {optimizer.param_groups[0]['lr']}"
        )

        if stopping_criterion and old_loss is not None:
            if abs(old_loss - train_loss) / old_loss < tol:
                print("Converged")
                break
            old_loss = train_loss
        elif stopping_criterion:
            old_loss = train_loss

    print("Done")
    print("Resampled atoms:", resamp_atom)
    return train_losses, list_D, times
