import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray
import jax.numpy as jnp
import typing


class ConvolutionalNeuralNetwork(eqx.Module):
    conv_1: eqx.nn.Conv2d
    conv_2: eqx.nn.Conv2d
    dense_1: eqx.nn.Linear
    dense_2: eqx.nn.Linear

    def __init__(self, num_classes: int , rng_key: PRNGKeyArray):
        conv_1_rng_key, conv_2_rng_key, layer_1_key, layer_2_key = jax.random.split(rng_key, 4)
        self.conv_1 = eqx.nn.Conv2d(1, 16, kernel_size=(3, 3), padding="SAME", key=conv_1_rng_key)
        self.conv_2 = eqx.nn.Conv2d(16, 64, kernel_size=(3, 3), padding="Same", key=conv_2_rng_key)

        self.dense_1 = eqx.nn.Linear(64, 256, key=layer_1_key)
        self.dense_2 = eqx.nn.Linear(256, num_classes, key=layer_1_key)

    def __call__(self, x: Array):
        x = jax.nn.relu(self.conv_1(x))
        x = eqx.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        x = jax.nn.relu(self.conv_2(x))
        x = eqx.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        # Global Average pooling
        x = jnp.mean(x, axis=(-2, -1))
        
        x = jax.nn.leaky_relu(self.dense_1(x))
        return self.dense_2(x)

class TestTensors(typing.NamedTuple):
    img_small: Array
    img_medium: Array
    img_large: Array

    @classmethod
    def create_default(cls, key: PRNGKeyArray):
        return TestTensors(
            img_small = jax.random.uniform(key1, shape=(1, 32, 32)),
            img_medium = jax.random.uniform(key2, shape=(1, 64, 64)),
            img_large = jax.random.uniform(key3, shape=(1, 224, 224))
        )


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model_key, key1, key2, key3 = jax.random.split(key, 4)

    model = ConvolutionalNeuralNetwork(num_classes=10, rng_key=model_key)

    # Test with 3 drastically different image resolutions:
    img_small = jax.random.uniform(key1, shape=(1, 32, 32))
    img_medium = jax.random.uniform(key2, shape=(1, 64, 64))
    img_large = jax.random.uniform(key3, shape=(1, 224, 224))

    out_small = model(img_small)
    out_medium = model(img_medium)
    out_large = model(img_large)

    print("Small Image (1, 32, 32)   -> Output Shape:", out_small.shape)
    print("Medium Image (1, 64, 64)  -> Output Shape:", out_medium.shape)
    print("Large Image (1, 224, 224) -> Output Shape:", out_large.shape)
