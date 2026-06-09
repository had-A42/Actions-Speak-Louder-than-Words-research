from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn

from .config import C_TASK_LABEL, E_TASK_LABEL
from .metrics import compute_normalized_entropy


def to_device(obj, device: torch.device | str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)
    return obj


def compute_multitask_loss(
    logits: torch.Tensor, labels: dict[str, torch.Tensor]
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss()
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)

    if logits.shape[1] == 1:
        return criterion(logits[:, 0], labels[C_TASK_LABEL].float())
    if logits.shape[1] == 2:
        loss_e = criterion(logits[:, 0], labels[E_TASK_LABEL].float())
        loss_c = criterion(logits[:, 1], labels[C_TASK_LABEL].float())
        return 0.5 * (loss_e + loss_c)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    train_log_every=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def evaluate():
        model.eval()

        all_like_labels = []
        all_full_labels = []

        all_like_logits = []
        all_full_logits = []

        all_like_scores = []
        all_full_scores = []

        with torch.no_grad():
            for batch in test_loader:
                batch = to_device(batch, device)

                logits = model(batch)

                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)

                if logits.shape[1] != 2:
                    raise ValueError(
                        f"Unexpected logits shape: {logits.shape}, need (batch_size, 2)"
                    )

                like_logits = logits[:, 0]
                full_logits = logits[:, 1]

                like_scores = torch.sigmoid(like_logits)
                full_scores = torch.sigmoid(full_logits)

                all_like_labels.append(
                    batch["labels"]["is_like"].detach().cpu().numpy()
                )
                all_full_labels.append(
                    batch["labels"]["is_full_play"].detach().cpu().numpy()
                )

                all_like_logits.append(like_logits.detach().cpu().numpy())
                all_full_logits.append(full_logits.detach().cpu().numpy())

                all_like_scores.append(like_scores.detach().cpu().numpy())
                all_full_scores.append(full_scores.detach().cpu().numpy())

        like_labels = np.concatenate(all_like_labels)
        full_labels = np.concatenate(all_full_labels)

        like_logits = np.concatenate(all_like_logits)
        full_logits = np.concatenate(all_full_logits)

        like_scores = np.concatenate(all_like_scores)
        full_scores = np.concatenate(all_full_scores)

        metrics = {
            "ne_e_task": compute_normalized_entropy(
                labels=like_labels,
                logits=like_logits,
            ),
            "ne_c_task": compute_normalized_entropy(
                labels=full_labels,
                logits=full_logits,
            ),
        }

        return metrics

    metrics = None

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_like_loss = 0.0
        running_full_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            if logits.shape[1] != 2:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            like_logits = logits[:, 0]
            full_logits = logits[:, 1]

            like_targets = batch["labels"]["is_like"].float()
            full_targets = batch["labels"]["is_full_play"].float()

            loss_like = criterion(like_logits, like_targets)
            loss_full = criterion(full_logits, full_targets)

            loss = 0.5 * loss_like + 0.5 * loss_full

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_like_loss += loss_like.item()
            running_full_loss += loss_full.item()
            n_steps += 1

            if step % train_log_every == 0:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"loss={running_loss / n_steps:.6f} "
                    f"loss_like={running_like_loss / n_steps:.6f} "
                    f"loss_full={running_full_loss / n_steps:.6f}"
                )

        metrics = evaluate()

        print(
            f"epoch={epoch + 1} "
            f"loss={running_loss / max(n_steps, 1):.6f} "
            f"loss_like={running_like_loss / max(n_steps, 1):.6f} "
            f"loss_full={running_full_loss / max(n_steps, 1):.6f} "
            f"metrics={metrics}"
        )

    return metrics
