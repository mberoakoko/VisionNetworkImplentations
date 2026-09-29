import time
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from data.dataloader import MNIST_Loader
from models.plain_vs_highway_no_pytree import ExperimentConfig


@dataclass
class TrainState: 
    params: dict
    opt_state: optax.OptState 

class TrainingHistory(typing.TypedDict):
    train_loss: list[float]
    train_acc: list[float]
    test_acc: list[float]
    grad_norms: list[float]

type RetTrainStep_t = tuple[TrainState, jax.Array, jax.Array, jax.Array]
type TranStep_t = typing.Callable[[TrainState, jax.Array, jax.Array],RetTrainStep_t ]


def compute_loss_and_accuracy(
    params: dict[str, typing.Any], 
    forward_fn: typing.Callable, 
    x: jax.Array, 
    y: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Computes cross-entropy loss and classification accuracy."""
    logits = forward_fn(params, x)
    one_hot = jax.nn.one_hot(y, num_classes=logits.shape[-1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y)
    return loss, acc


def create_train_step(foward_func: typing.Callable, optimizer: optax.GradientTransformation) -> TranStep_t:
    
    def loss_function(params: dict[str, typing.Any], x: jax.Array, y: jax.Array) -> jax.Array: 
        logits = foward_func(params, x)
        one_hot = jax.nn.one_hot(y, num_classes=logits.shape[-1])
        return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))



    @jax.jit
    def train_step(state: TrainState, x: jax.Array, y: jax.Array) -> RetTrainStep_t:
        loss_val, grads = jax.value_and_grad(loss_function)(state.params, x, y)

        # Calculate gradient norm across all parameters
        grad_flat, _ = jax.tree_util.tree_flatten(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grad_flat))

        # Calculate predictions accuracy
        logits = foward_func(state.params, x)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)

        # Optax parameter update step
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        return TrainState(params=new_params, opt_state=new_opt_state), loss_val, acc, grad_norm

    return train_step 


def run_experiment(
        model_name: str,
        forward_func: typing.Callable,
        init_fn: typing.Callable,
    dataloader_obj: MNIST_Loader,
    config: ExperimentConfig
) -> TrainingHistory:
    print("\n==================================================")
    print(f" Starting Experiment: {model_name} ({config.num_layers} Deep Layers)")
    print("==================================================")
    
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)


    params = init_fn(init_key, config)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    state = TrainState(params=params, opt_state=opt_state)

    train_step_fn = create_train_step(forward_func, optimizer)

    history: TrainingHistory = {
        "train_loss" : [],
        "test_acc": [],
        "grad_norms": [],
        "train_acc":  []
    }

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        key, epoch_key = jax.random.split(key)

        # initialize losses before accumuclation 
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader_obj.get_batches(epoch_key, config.batch_size, split="train"):
            state, loss_fn, acc_val , g_norm = train_step_fn(state, batch_x, batch_y)

            epoch_loss += float(loss_fn)
            epoch_acc += float(acc_val)
            epoch_grad_norm += float(g_norm)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        avg_grad_norm = epoch_grad_norm / num_batches

        # Evaluation on test set
        test_accs: list[float] = []
        for bx, by in dataloader_obj.get_batches(epoch_key, config.batch_size, split="test"):
            _, t_acc = compute_loss_and_accuracy(state.params, forward_func, bx, by)
            test_accs.append(float(t_acc))
        avg_test_acc = float(jnp.mean(test_accs))

        epoch_time = time.time() - start_time

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)
        history["test_acc"].append(avg_test_acc)
        history["grad_norms"].append(avg_grad_norm)

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} [{epoch_time:.2f}s] | "
            f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc*100:.2f}% | "
            f"Test Acc: {avg_test_acc*100:.2f}% | Grad Norm: {avg_grad_norm:.4f}"
        )

        return history 
