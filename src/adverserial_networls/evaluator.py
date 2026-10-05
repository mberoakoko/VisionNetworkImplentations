from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.data.grain_data_loader import GrainDatasetAdapter, ImageBatch


class EvalMetrics(NamedTuple):
    accuracy: float
    avg_loss: float
    total_samples: int
    correct_count: int


# 1. Pure batch metrics computer
@jax.jit
def _evaluate_step(
    model: eqx.Module, images: jnp.ndarray, labels: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Computes total correct count, total batch loss, and batch size for a single batch."""
    logits = jax.vmap(model)(images)

    # Sum of cross-entropy losses in the batch
    batch_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )

    preds = jnp.argmax(logits, axis=-1)
    correct_count = jnp.sum(preds == labels)
    batch_size = labels.shape[0]

    return batch_loss, correct_count, batch_size


# 2. Dataset-wide evaluator
def evaluate_dataset(
    model: eqx.Module,
    test_loader,
    pgd_attack_fn: Optional[callable] = None,
    epsilon: float = 0.1,
) -> EvalMetrics:
    """Iterates over the entire test dataset loader and computes global accuracy and loss.

    Args:
        model: Equinox CNN module.
        test_loader: Grain test dataset iterator.
        pgd_attack_fn: Optional PGD attack function to evaluate adversarial accuracy.
        epsilon: Radius of L-infinity perturbation if pgd_attack_fn is provided.
    """
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for batch in GrainDatasetAdapter(test_loader):
        images, labels = batch.images, batch.labels

        # Apply PGD attack if evaluating under adversarial perturbation
        if pgd_attack_fn is not None:
            images = pgd_attack_fn(model, images, labels, epsilon=epsilon)

        batch_loss, correct, size = _evaluate_step(model, images, labels)

        total_loss += float(batch_loss)
        total_correct += int(correct)
        total_samples += int(size)

    global_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    global_avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return EvalMetrics(
        accuracy=global_accuracy,
        avg_loss=global_avg_loss,
        total_samples=total_samples,
        correct_count=total_correct,
    )
