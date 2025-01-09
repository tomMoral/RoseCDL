
import torch
from tqdm import tqdm


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
        X_hat, z_hat = model(X)
        if hasattr(loss_fn, "get_outliers_mask"):
            outliers_mask = loss_fn.get_outliers_mask(X_hat, z_hat, X)
            loss = loss_fn(X_hat, z_hat, X, outliers_mask=outliers_mask)
        else:
            loss = loss_fn(X_hat, z_hat, X)

        avg_loss += loss.item()
        count += 1

        def closure():
            with torch.no_grad():
                X_hat, z_hat = model(X)
                if hasattr(loss_fn, "get_outliers_mask"):
                    return loss_fn(X_hat, z_hat, X, outliers_mask=outliers_mask)
                return loss_fn(X_hat, z_hat, X)

        # Backpropagation
        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()
        model.rescale()

        if scheduler is not None:
            scheduler.step()

        if max_batch is not None and batch >= max_batch:
            break

    return (avg_loss / count)


def train(
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    scheduler=None,
    epochs=10,
    max_batch=None,
    callbacks=(),
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

    print("TRAINING STARTED")

    old_loss = None
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        # Compute thresholds for outliers detection
        if hasattr(loss_fn, "compute_outlier_threshold"):
            loss_fn.compute_outlier_thresholds(model, train_dataloader)

        train_loss = train_loop(
            train_dataloader,
            model.csc,
            loss_fn,
            optimizer,
            max_batch=max_batch,
            scheduler=scheduler,
        )

        for callback in callbacks:
            callback(model, epoch, train_loss)

        pbar.set_description(
            f"Epoch {epoch+1}"
            f" - Average train loss: {train_loss:.15f}"
            f" - Step size: {optimizer.param_groups[0]['lr']}"
        )

        if stopping_criterion:
            if old_loss is not None and abs(old_loss - train_loss) / old_loss < tol:
                print("Converged")
                break
            old_loss = train_loss

        # resample atoms
        model.csc.resample_atom()
