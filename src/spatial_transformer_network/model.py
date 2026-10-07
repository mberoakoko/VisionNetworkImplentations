import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class SpatialTransformerNetwork(eqx.Module):
    out_shape: tuple[int, int]

    def __init__(self, out_shape: tuple[int, int]):
        self.out_shape = out_shape

    def __call__(self, u: Float[Array, "Channel H_in W_in"], theta: Array) -> Array:
        c, h_in, w_in = u.shape
        h_out, w_out = self.out_shape

        y_t, x_t = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, h_out),
            jnp.linspace(-1.0, 1.0, w_out),
            indexing="ij"
        )

        grid_target = jnp.stack([
            x_t.flatten(),
            y_t .flatten(),
            jnp.ones_like(x_t.flatten())
        ])

        grid_source = theta @ grid_target

        x_s = grid_source[0, :].reshape(h_out, w_out)
        y_s = grid_source[0, :].reshape(h_out, w_out)

        # Normalization of the coordinate system
        x_s = (x_s + 1.0) * (w_in - 1) / 2.0
        y_s = (y_s + 1.0) * (h_in - 1) / 2.0

        coords = jnp.stack([y_s, x_s], axis=0)

        # this is where we perform the blinear interpolation
        def sample_channel(channel: Array) -> jax.Array:
            return jax.scipy.ndimage.map_coordinates(
                channel,
                coords,
                order=1,  # order=1 specifies bilinear interpolation
                mode="constant",  # values outside bounds map to cval
                cval=0.0,
            )

        return jax.vmap(sample_channel)(u)


