import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray


# 1. Batched Accuracy Function
@jax.jit
def accuracy(model: eqx.Module, data: tuple[Array, Array]) -> Array:
    inputs, labels = data
    # Vectorize the model over the batch dimension
    logits = jax.vmap(model)(inputs)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


# 2. Loss Function with L2 Regularization
def loss_fun(model: eqx.Module, l2reg: float, data: tuple[Array, Array]) -> Array:
    """Compute cross-entropy loss + L2 regularization for Equinox model."""
    inputs, labels = data
    x = inputs.astype(jnp.float32)

    # Batched forward pass
    logits = jax.vmap(model)(x)
    loss_value = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

    if l2reg > 0.0:
        # Filter leaves to get only array parameters (excluding static metadata if any)
        params = eqx.filter(model, eqx.is_array)
        # Compute L2 squared norm across all parameter leaves in the PyTree
        sqnorm = sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return loss_value + 0.5 * l2reg * sqnorm

    return loss_value


# 3. PGD Adversarial Attack
@jax.jit
def pgd_attack(
    model: eqx.Module,
    image: Array,
    label: Array,
    epsilon: float = 0.1,
    maxiter: int = 10,
) -> Array:
    """PGD attack on a single image or batch for Equinox models.

    Args:
        model: Equinox Module.
        image: Input tensor of shape (C, H, W) or (B, C, H, W).
        label: Target label array.
        epsilon: Radius of L-infinity perturbation ball.
        maxiter: Attack iterations.
    """
    image_perturbation = jnp.zeros_like(image)

    def adversarial_loss(perturbation: Array) -> Array:
        # Pass (image + perturbation) through loss_fun with l2reg=0
        return loss_fun(model, 0.0, (image + perturbation, label))

    # Gradient of loss with respect to perturbation (argnums=0)
    grad_adversarial = jax.grad(adversarial_loss)

    step_size = 2 * epsilon / maxiter

    def step_fn(i, pert):
        # Compute gradient wrt image perturbation
        grad = grad_adversarial(pert)
        # Gradient ascent step using sign of gradient
        pert = pert + step_size * jnp.sign(grad)
        # Projection onto L-infinity ball centered at image
        return jnp.clip(pert, -epsilon, epsilon)

    # Use jax.lax.fori_loop for JIT-compatible static loop iteration
    image_perturbation = jax.lax.fori_loop(0, maxiter, step_fn, image_perturbation)

    # Clip final image to valid pixel range [0, 1]
    return jnp.clip(image + image_perturbation, 0.0, 1.0)