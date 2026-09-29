
import typing
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from loguru import logger
from matplotlib.figure import Figure

plt.style.use("bmh")
plt.rcParams.update({"font.size": 8})


@dataclass
class ExperimentConfig:
    min_val: int = -3
    max_val: int = 3
    learning_rate: float = 1e-3
    num_epochs: int  = 10000

class TrainingData(typing.NamedTuple):
    train_data: tuple[jax.Array, jax.Array]
    test_data: tuple[jax.Array, jax.Array]


def generate_training_data(n_samples: int, config: ExperimentConfig) -> TrainingData: 
    _KEY = jax.random.PRNGKey(0)
    _KEY, x_key, y_noise_key_train, y_nosie_key_test = jax.random.split(_KEY, 4)
    x_samples = jax.random.uniform(x_key, shape=(n_samples, 1), minval=config.min_val, maxval=config.max_val)
    return TrainingData(
        train_data=(x_samples, jnp.sin(2 * jnp.pi * 2 * x_samples) + jax.random.normal(y_noise_key_train, (n_samples, 1)) * 0.2 ),
        test_data=(x_samples, jnp.sin(2 * jnp.pi * 2 * x_samples) + jax.random.normal(y_nosie_key_test, (n_samples, 1)) * 0.2)
    )


class SimpleNeuralNetwork(eqx.Module):
    layers: list[eqx.nn.Linear]

    def __init__(self, layer_sizes: list[int], key: jax.Array): 
        self.layers = []

        for (fan_in, fan_out ) in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, sub_key = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(fan_in, fan_out, key=sub_key))


    def __call__(self, x: jax.Array):
        a = x 
        for layer in self.layers[:-1]:
            a = jax.nn.gelu(layer(a))
        a = self.layers[-1](a)
        return a 

def train_model(model: SimpleNeuralNetwork, x: jax.Array, y: jax.Array, config: ExperimentConfig | None = None) -> SimpleNeuralNetwork: 
    def model_to_loss(model_: SimpleNeuralNetwork, x_: jax.Array, y_: jax.Array) -> jax.Array:
        pred = jax.vmap(model_)(x_)
        error = pred - y_
        return jnp.mean(error ** 2)

    def model_to_loss_and_grad():
        return eqx.filter_value_and_grad(model_to_loss)
    
    @eqx.filter_jit 
    def make_step(model_: SimpleNeuralNetwork, optimizer: typing.Any, opt_state_: typing.Any, x_: jax.Array, y_: jax.Array) -> tuple[SimpleNeuralNetwork, typing.Any, jax.Array]: 
        loss, grad = model_to_loss_and_grad()(model_,  x_, y_)
        updates, opt_state_ = optimizer.update(grad, opt_state_, model_)
        model_  = eqx.apply_updates(model_, updates)
        return model_, opt_state_, loss 

    config: ExperimentConfig = config or ExperimentConfig()
    
    optimizer = optax.sgd(learning_rate=config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    logger.debug(f"Optimizer State => {opt_state}")
    
    loss_history = []
    for epoch in range(config.num_epochs):
        model, opt_state, loss = make_step(model, optimizer, opt_state, x, y)
        loss_history.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss {loss}", end="\r")
    
    logger.debug("Finished training run ")
    return model 

        


def main() -> None:
    key = jax.random.PRNGKey(1)
    train_data, test_data = generate_training_data(n_samples=50000, config=ExperimentConfig())

    layers = [1, 100, 50, 50, 100 , 1]
    model = SimpleNeuralNetwork(layers, key=key)
    

    model = train_model(model, train_data[0], train_data[1])
    
    vectorized_model = jax.vmap(model)
    fig: Figure = plt.figure(figsize=(16 , 9))
    y = vectorized_model(test_data[0])
    logger.debug(f"{y.shape=}")
    plt.scatter(test_data[0], test_data[1], c="C1")
    plt.scatter(test_data[0], y, c="C2")
    plt.tight_layout()
    logger.info("Plotting result")
    plt.show()

 


if __name__ == "__main__":
    main()
    

    
