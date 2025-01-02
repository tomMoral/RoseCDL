import time

import torch
from tqdm import tqdm

from .utils_outliers import compute_error, get_outlier_mask, get_thresholds


def compute_objective(X, X_hat, z_hat, reg):
    loss_fn = torch.nn.MSELoss(reduction="sum")
    loss = loss_fn(X, X_hat) / (2 * X.shape[0])
    loss += reg * (z_hat.sum() / z_hat.shape[0]).item()
    return loss


def compute_loss(
    model,
    loss_fn,
    X,
    thresholds=None,
    outliers_mask=None,
    moving_average=None,
    union_channels=True,
    return_n_outliers=False,
    return_outliers_mask=False,
    adapt_alphacsc=False,
    add_reg=True,
    opening_window=True,
    per_patch=True,
):
    """

    moving_average : dict, optional
        Moving average parameters, by default None
        example: moving_average=dict(
            window_size=int(model.n_times_atom),
            method='max',  # 'max', 'average' or 'gaussian'
        )

    """
    if thresholds is not None:
        # XXX: put lmbd to 0 for outliers detection
        prediction = model(X)

        if outliers_mask is None:
            # Compute error vector, keep it 3D
            err = compute_error(
                prediction=prediction,
                X=X,
                loss_fn=loss_fn,
                per_patch=(
                    model.n_times_atom if per_patch else False
                ),  # compute error per patch
                device=model.device,
                z_hat=model.z_hat_.clone() if add_reg else None,
                lmbd=model.lmbd,
            )
            # Remove outliers
            outliers_mask = get_outlier_mask(
                data=err,
                thresholds=thresholds,
                moving_average=moving_average,
                opening_window=model.n_times_atom if opening_window else None,
                union_channels=union_channels,
            )

        if outliers_mask.ndim == 2 and X.ndim == 3:
            # Duplicate across channels
            outliers_mask = outliers_mask.unsqueeze(1).expand_as(X)

        assert (
            outliers_mask.shape == X.shape
        ), f"outliers_mask.shape: {outliers_mask.shape}, X.shape: {X.shape}"

        n_outliers = outliers_mask.sum().item()
        # Compute loss
        loss = loss_fn(prediction[~outliers_mask], X[~outliers_mask])
    else:
        # Compute prediction and loss
        loss = loss_fn(model(X), X)
        n_outliers = None

    if adapt_alphacsc:
        # Add regularization so that the loss is comparable to alphaCSC
        loss = 0.5 * loss + model.lmbd * model.z_hat_.sum().item()

    if return_n_outliers:
        if return_outliers_mask:
            return loss, n_outliers, outliers_mask
        else:
            return loss, n_outliers
    else:
        if return_outliers_mask:
            return loss, outliers_mask
        return loss


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    max_batch=None,
    scheduler=None,
    thresholds=None,
    moving_average=None,
    union_channels=True,
    add_reg=True,
    opening_window=True,
    per_patch=True,
):
    avg_loss = 0
    count = 0
    for batch, X in enumerate(dataloader):
        loss, n_outliers, outliers_mask = compute_loss(
            model,
            loss_fn,
            X,
            return_n_outliers=True,
            return_outliers_mask=True,
            thresholds=thresholds,
            moving_average=moving_average,
            union_channels=union_channels,
            add_reg=add_reg,
            opening_window=opening_window,
            per_patch=per_patch,
        )

        avg_loss += loss.item()
        count += 1

        def closure():
            with torch.no_grad():
                return compute_loss(
                    model,
                    loss_fn,
                    X,
                    thresholds=thresholds,
                    outliers_mask=outliers_mask,
                    moving_average=moving_average,
                    union_channels=union_channels,
                    add_reg=add_reg,
                    opening_window=opening_window,
                    per_patch=per_patch,
                )

        # Backpropagation
        loss.backward()
        optimizer.step(closure)
        optimizer.zero_grad()
        model.rescale()

        if scheduler is not None:
            scheduler.step()

        if max_batch is not None and batch >= max_batch:
            break

    return (avg_loss / count), n_outliers


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
    outliers_kwargs=None,
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

    train_losses = []
    list_D = []
    times = []
    outliers = []

    if save_list_D:
        if model.rank == "full":
            list_D.append(model.D_hat_)
        elif model.rank == "uv_constraint":
            list_D.append(model.uv_hat_)
        # Get first batch and compute init loss
        for X in train_dataloader:
            init_loss = compute_loss(model, loss_fn, X)
            break
        train_losses.append(init_loss.item())
        start = time.time()
        times.append(0)
    pbar = tqdm(range(epochs))

    if outliers_kwargs is not None:
        outliers_kwargs = outliers_kwargs.copy()
        moving_average = outliers_kwargs.pop("moving_average", None)
        union_channels = outliers_kwargs.pop("union_channels", True)
        add_reg = outliers_kwargs.pop("add_reg", False)
        opening_window = outliers_kwargs.pop("opening_window", True)
        per_patch = outliers_kwargs.pop("per_patch", True)
        # By default, outliers detection is performed from the first epoch
        first_outlier_epoch = outliers_kwargs.pop("first_outlier_epoch", 0)
    else:
        moving_average = None
        union_channels = None
        add_reg = None
        opening_window = None
        per_patch = False

    resamp_atom = []
    resamp_iter = []

    for epoch in pbar:
        # Compute thresholds for outliers detection
        if outliers_kwargs is not None and epoch >= first_outlier_epoch:
            err_i_batches = []
            n_samples = 0
            for X in train_dataloader:
                X_hat = model(X)
                err_i = compute_error(
                    prediction=X_hat,
                    X=X,
                    loss_fn=loss_fn,
                    per_patch=model.n_times_atom if per_patch else False,
                    device=model.device,
                    z_hat=model.z_hat_.clone() if add_reg else None,
                    lmbd=model.lmbd,
                )
                err_i_batches.append(err_i.flatten())

                n_samples += err_i.size(0)
                if n_samples >= 100:
                    # Compute thresholds on minimum 100 samples
                    break

            err = torch.cat(err_i_batches)
            thresholds = get_thresholds(err, **outliers_kwargs)
        else:
            thresholds = None

        train_loss, n_outliers = train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            max_batch=max_batch,
            scheduler=scheduler,
            thresholds=thresholds,
            moving_average=moving_average,
            union_channels=union_channels,
            add_reg=add_reg,
            opening_window=opening_window,
            per_patch=per_patch,
        )

        train_losses.append(train_loss)
        outliers.append(n_outliers)

        if save_list_D:
            times.append(time.time() - start)
            if model.rank == "full":
                list_D.append(model.D_hat_)
            elif model.rank == "uv_constraint":
                list_D.append(model.uv_hat_)

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

        # resample atoms
        with torch.no_grad():
            # compute sparsity, resample unsed atom if needed
            z_nnz = torch.sum(model.z_hat_ != 0, axis=(0, 2))
            null_atom_indices = torch.where(z_nnz < 2)[0]
            if len(null_atom_indices) > 0:
                # resample a random atom
                idx = torch.randint(0, len(null_atom_indices), (1,))
                k0 = null_atom_indices[idx].item()
                # no resampling of the last 2 resampled atoms
                if k0 in resamp_atom[-2:]:
                    continue
                # Create a new random atom and rescale
                model.resample_atom(k0)

                model.rescale()
                resamp_atom.append(k0)
                resamp_iter.append(epoch)

    train_hist = dict(
        train_losses=train_losses,
        list_D=list_D,
        times=times,
        outliers=outliers,
        resamp_atom=resamp_atom,
        resamp_iter=resamp_iter,
    )

    return train_hist
