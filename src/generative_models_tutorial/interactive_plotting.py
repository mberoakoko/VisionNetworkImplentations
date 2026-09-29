import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array

# Import the math functions from your main pipeline
# Adjust 'your_module_name' to the actual name of your file (e.g., ppca_model)
from generative_models_tutorial.ppca_pipeline import (
    decode,
    encode,
    load_mnist_images,
    load_mnist_labels,
    ppca_em_step_fit,
)


class InteractivePPCADashboard:
    def __init__(self, Z: Array, y: np.ndarray, W: Array, mu: Array):
        self.W = W
        self.mu = mu
        
        # Apply a high-contrast dark theme for the dashboard
        # plt.style.use("dark_background")
        
        self.fig, (self.ax_img, self.ax_scatter) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.canvas.manager.set_window_title("PPCA Latent Space Explorer")
        
        # Right Side: Latent Space Scatter Plot
        self.scatter = self.ax_scatter.scatter(
            Z[:, 0], Z[:, 1], c=y, cmap="tab10", alpha=0.7, s=10, edgecolors="none"
        )
        self.ax_scatter.set_title("Latent Space Manifold (Click to decode)")
        self.ax_scatter.set_xlabel("Latent Dimension 1 (Z0)")
        self.ax_scatter.set_ylabel("Latent Dimension 2 (Z1)")
        self.ax_scatter.grid(True, color="#333333", linestyle="--", alpha=0.5)
        self.fig.colorbar(self.scatter, ax=self.ax_scatter, label="Digit Class")
        
        # Track the currently selected coordinate with a distinct marker
        self.cursor_marker, = self.ax_scatter.plot(
            0, 0, marker="+", color="white", markersize=15, markeredgewidth=2
        )
        
        # Left Side: Decoded Image
        initial_z = jnp.array([[0.0, 0.0]])
        initial_img = self._get_decoded_image(initial_z)
        
        self.im = self.ax_img.imshow(initial_img, cmap="magma", vmin=0, vmax=1)
        self.ax_img.set_title("Decoded Output: Z = [0.00, 0.00]")
        self.ax_img.axis("off")
        
        # Register the click event listener
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        
        plt.tight_layout()
        plt.show()

    def _get_decoded_image(self, z: Array) -> np.ndarray:
        """Pushes the (1, 2) Z vector through the JAX decoder and reshapes."""
        x_recon = decode(z, self.W, self.mu)
        return np.clip(x_recon[0], 0, 1).reshape(28, 28) * 255.0

    def onclick(self, event):
        """Callback for Matplotlib canvas clicks."""
        # Ignore clicks that are outside the bounds of the scatter plot axes
        if event.inaxes != self.ax_scatter:
            return
            
        # Extract the precise (x, y) float coordinates from the click
        zx, zy = event.xdata, event.ydata
        z_click = jnp.array([[zx, zy]])
        
        
        # Decode the coordinate into pixel space
        new_img = self._get_decoded_image(z_click)
        print(new_img, end="\r")
        
        # Update the UI components
        self.im.set_data(new_img)
        self.ax_img.set_title(f"Decoded Output: Z = [{zx:.2f}, {zy:.2f}]")
        self.cursor_marker.set_data([zx], [zy])
        
        # Trigger a UI repaint without blocking
        self.fig.canvas.draw_idle()

def launch_dashboard(X_sub: np.ndarray, y_sub: np.ndarray, W: Array, sigma2: float, mu: Array):
    """Entry point to encode data and boot the interactive Matplotlib loop."""
    print("Encoding dataset for dashboard visualization...")
    Z = encode(jnp.array(X_sub), W, sigma2, mu)
    
    print("Launching interactive UI...")
    InteractivePPCADashboard(Z, y_sub, W, mu)

if __name__ == "__main__":
    # Example integration:
    # Assuming you have your state_result and data from your main file
    # from your_module_name import load_mnist_images, load_mnist_labels, ppca_em_step_fit
    
    x_images = load_mnist_images()
    y = load_mnist_labels()
    x_all = x_images.reshape(-1, 28*28).astype(np.float32) / 255.0 
    
    N_samples = 15000
    X_sub = x_all[:N_samples, :]
    y_sub = y[:N_samples]
    
    state_result = ppca_em_step_fit(jnp.array(x_all), max_iters=150)
    
    launch_dashboard(X_sub, y_sub, state_result.w, state_result.sigma, state_result.mu)
