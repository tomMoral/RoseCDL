import numpy as np
import torch
import time

from tqdm import tqdm

from .utils import get_z_nnz


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    max_batch=None,
    scheduler=None
):
    avg_loss = 0
    count = 0
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        loss = loss_fn(model(X), X)

        avg_loss += loss.item()
        count += 1

        def closure():
            with torch.no_grad():
                return loss_fn(model(X), X)

        # Backpropagation

        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()
        model.rescale()

        if False:
            # compute sparsity, resample unsed atom if needed
            z_nnz = get_z_nnz(model.z.to("cpu").detach().numpy())
            null_atom_indices = np.where(z_nnz == 0)[0]
            if len(null_atom_indices) > 0:
                k0 = null_atom_indices[0]  # only the first one? why so?
                model.resample_atom(k0, X)

        if scheduler is not None:
            scheduler.step()

        if max_batch is not None and batch >= max_batch:
            break

    return avg_loss / count


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
    tol=1e-8
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
    pbar = tqdm(range(epochs))

    for epoch in pbar:

        train_loss = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            max_batch=max_batch,
            scheduler=scheduler
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
    return train_losses, list_D, times
