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
    print(f"{x.shape=}")
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


def compute_loss(model: eqx.Module, images: Array, labels: Array) -> Array:
    """Computes mean cross-entropy loss over a batch (B, 1, H, W)."""
    logits = jax.vmap(model)(images)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


# ---------------------------------------------------------------------------
# 2. Single-Step Functional Update Routines
# ---------------------------------------------------------------------------


@jax.jit(static_argnames="optimizer")
def standard_train_step(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    images: Array,
    labels: Array,
) -> tuple[eqx.Module, optax.OptState, Array]:
    """Executes one standard training step over clean images."""
    loss_val, grads = jax.value_and_grad(compute_loss)(model, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


def adversarial_train_step_builder(pgd_attack_fn):
    """Factory creating a JIT-compiled adversarial training step function."""

    @jax.jit(static_argnames="optimizer")
    def adv_train_step(
        model: eqx.Module,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        images: Array,
        labels: Array,
        epsilon: float = 0.1,
    ) -> tuple[eqx.Module, optax.OptState, Array]:
        """Generates PGD adversarial examples dynamically, then updates model parameters."""
        # 1. Generate adversarial images on-the-fly
        adv_images = pgd_attack_fn(model, images, labels, epsilon=epsilon)

        # 2. Compute gradients on adversarial samples
        loss_val, grads = jax.value_and_grad(compute_loss)(model, adv_images, labels)

        # 3. Apply updates
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    return adv_train_step


# ---------------------------------------------------------------------------
# 3. Evaluation & Metrics
# ---------------------------------------------------------------------------


@jax.jit
def evaluate_batch(
    model: eqx.Module, images: Array, labels: Array
) -> tuple[Array, Array]:
    """Computes loss and accuracy for an image batch without tracking gradients."""
    logits = jax.vmap(model)(images)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, acc


# ---------------------------------------------------------------------------
# 4. High-Level Training Epoch Loop
# ---------------------------------------------------------------------------


def run_training_epoch(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    dataset_generator,
    step_fn,
    num_steps: int = 50,
) -> tuple[eqx.Module, optax.OptState, float]:
    """Runs a single epoch training loop using the specified step_fn."""
    total_loss = 0.0

    for step in range(1, num_steps + 1):
        # Fetch next synthetic data batch (B, 1, H, W)
        batch = dataset_generator()
        model, opt_state, loss_val = step_fn(
            model, opt_state, optimizer, batch.images, batch.labels
        )
        total_loss += float(loss_val)

    avg_loss = total_loss / num_steps
    return model, opt_state, avg_loss