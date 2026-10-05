import typing

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


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


class ImageBatch(typing.NamedTuple):
    images: Array  # Shape: (batch_size, 1, H, W)
    labels: Array  # Shape: (batch_size,)


class TestDataGenerator(typing.NamedTuple):
    small: ImageBatch
    medium: ImageBatch
    large: ImageBatch

    @classmethod
    def create(
        cls,
        key: PRNGKeyArray,
        batch_size: int = 1,
        num_classes: int = 10,
    ) -> "TestDataGenerator":
        """Generates mock image batches and integer labels for small, medium, and large resolutions."""
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        def make_batch(img_key: PRNGKeyArray, lbl_key: PRNGKeyArray, height: int, width: int) -> ImageBatch:
            images = jax.random.uniform(img_key, shape=(batch_size, 1, height, width), minval=0.0, maxval=1.0)
            labels = jax.random.randint(lbl_key, shape=(batch_size,), minval=0, maxval=num_classes)
            return ImageBatch(images=images, labels=labels)

        return TestDataGenerator(
            small=make_batch(k1, k2, 32, 32),
            medium=make_batch(k3, k4, 64, 64),
            large=make_batch(k5, k6, 224, 224),
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
