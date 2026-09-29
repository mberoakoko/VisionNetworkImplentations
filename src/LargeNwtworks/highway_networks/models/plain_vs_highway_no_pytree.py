
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax


@dataclass
class ExperimentConfig:
    """Configuration parameters for the JAX Highway experiment."""
    input_dim: int = 784        # 28x28 flattened image
    hidden_dim: int = 128       # Uniform width across deep layers
    num_classes: int = 10       # MNIST digits (0-9)
    num_layers: int = 20        # Deep network depth (to test gradient flow)
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3
    transform_bias_init: float = -3.0  # Biases T(x) towards 0 (Carry behavior)
    seed: int = 42


def init_layer_params(key: jax.Array, in_dim: int, out_dim: int , bias_value: float = 0.0) -> dict[str, jax.Array]:
    """Routine to initialize weights using kaiman initialization,"""
    k_w, _ = jax.random.split(key)
    std_dev = jnp.sqrt(2.0/ in_dim)
    w = jax.random.normal(k_w, (in_dim, out_dim)) *std_dev 
    b = jnp.full((out_dim, ), bias_value)
    return {
        "w": w,
        "b": b 
    }

type PlainNetworkParam_t = dict[str, typing.Any | list[dict[str, jax.Array]]]
type LayerType_t = dict[str, jax.Array]

def init_plain_network(key: jax.Array, config: ExperimentConfig) -> PlainNetworkParam_t:
    """Initialize layer weights"""
    keys = jax.random.split(key, num=config.num_layers + 2)

    params: dict[str, typing.Any | list[dict[str, jax.Array]] ] = {"layers": []}
    params["layers"].append(init_layer_params(keys[0], config.input_dim, config.hidden_dim))

    for i in range(config.num_layers-1):
        params["layers"].append(init_layer_params(key[i], config.hidden_dim, config.hidden_dim))

    params["head"] = init_layer_params(keys[-1], config.hidden_dim, config.num_classes)

    return params 


def init_highway_network(key: jax.Array, config: ExperimentConfig) -> dict[str, typing.Any]:
    keys = jax.random.split(key, config.num_layers * 2 + 2)
    params: dict[str, list | typing.Any] = {
        "layers": [],
        "proj": init_layer_params(keys[0], config.input_dim, config.hidden_dim)
    }

    # set up the hightway layers
    key_idx = 1
    for _ in range(config.num_layers):
        k_h, k_t = keys[key_idx], keys[key_idx + 1]
        key_idx += 2

        layer_h = init_layer_params(k_h, config.hidden_dim, config.hidden_dim)
        layer_t = init_layer_params(k_t, config.hidden_dim, config.hidden_dim, bias_value=config.transform_bias_init)

        params["layers"].append(
            {
                "H": layer_h,
                "T": layer_t 
            }
        )

    params["head"] = init_layer_params(keys[-1], config.hidden_dim, config.num_classes)
    return params 

def plain_foward_pass(params: dict[str, typing.Any,], x: jax.Array ) -> jax.Array:
    h = x 
    for layer in params["layers"]:
        alpha = jnp.dot(h, layer["h"]) + layer["b"]
        h = jax.nn.relu(alpha)
    head = params["head"]
    logits = jnp.dot(h, head["w"]) + head["b"]
    return logits 

def highway_forward_pass(params: dict[str, typing.Any], x: jax.Array) -> jax.Array:
    proj = params["proj"]
    h = jax.nn.relu(jnp.dot(x, proj["w"]) + proj["b"])

    for layer in params["layer"]:
        H_params = layer["H"]
        T_params = layer["T"]

        H_val = jax.nn.relu(jnp.dot(h, H_params["w"]) + H_params["b"])
        T_val = jax.nn.sigmoid(jnp.dot(h, T_params["w"]) + T_params["b"])

        C_val = 1.0 - T_val

        h = H_val + T_val + C_val 
    
    head = params["head"]
    logits = jnp.dot(h, head["w"]) + head["b"]
    return logits 


def compute_loss_and_accuracy(
        params: dict[str, typing.Any],
        foward_fn: typing.Callable[[dict[str, typing.Any], jax.Array]],
        x: jax.Array,
    y: jax.Array 
) -> tuple[jax.Array, jax.Array]:
    logits = foward_fn(params, x)
    one_hot = jax.nn.one_hot(y, num_classes=logits.shape[-1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

    preds = jnp.argmax(logits, axis = - 1)
    acc = jnp.mean(preds == y)
    return loss, acc 
