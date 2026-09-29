import gzip
import pathlib

import jax.numpy as jnp
import numpy as np
from loguru import logger

# Assuming _ROOT_DATA_FOLDER is defined as in the previous script
_ROOT_DATA_FOLDER = pathlib.Path(__file__).parent / "store/mnist_data"
_TRAIN_IMAGES_PATH = _ROOT_DATA_FOLDER / "train-images-idx3-ubyte.gz"
_TRAIN_IMAGES_LABELS = _ROOT_DATA_FOLDER / "train-labels-idx1-ubyte.gz"


def load_mnist_images(filepath: pathlib.Path = _TRAIN_IMAGES_PATH) -> np.ndarray:
    """Reads MNIST image files and normalizes them to float32 [0, 1]."""
    logger.info(f"Reading from {filepath=}")
    with gzip.open(filepath, 'rb') as f:
        # Images have a 16-byte header before the pixel data starts
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    
    # Reshape to (N, H, W, Channels) which is standard for most convnets
    # Cast to float32 and normalize to [0.0, 1.0]
    return data.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

def load_mnist_labels(filepath: pathlib.Path = _TRAIN_IMAGES_LABELS) -> np.ndarray:
    """Reads MNIST label files."""
    logger.info(f"Reading from {filepath=}")
    with gzip.open(filepath, 'rb') as f:
        # Labels have an 8-byte header
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    
    return data.astype(np.int32)

def get_jax_dataloader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 42):
    """
    A lightweight generator that yields batches of JAX arrays.
    """
    num_samples = len(images)
    indices = np.arange(num_samples)
    
    # We use numpy's random generator on the CPU for shuffling. 
    # JAX's random keys are usually reserved for model initialization and dropout on the GPU.
    rng = np.random.default_rng(seed)
    
    if shuffle:
        rng.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Convert the CPU numpy arrays to Device JAX arrays just-in-time
        batch_images = jnp.array(images[batch_indices])
        batch_labels = jnp.array(labels[batch_indices])
        
        yield batch_images, batch_labels

# ==========================================
# Example usage in a JAX pipeline
# ==========================================
if __name__ == "__main__":
    # 1. Load data into CPU RAM
    train_images_path = _ROOT_DATA_FOLDER / "train-images-idx3-ubyte.gz"
    train_labels_path = _ROOT_DATA_FOLDER / "train-labels-idx1-ubyte.gz"
    
    print("Loading data into memory...")
    X_train = load_mnist_images(train_images_path)
    Y_train = load_mnist_labels(train_labels_path)
    
    print(f"Loaded {len(X_train)} images of shape {X_train.shape}")
    
    # 2. Instantiate the dataloader
    batch_size = 128
    train_loader = get_jax_dataloader(X_train, Y_train, batch_size=batch_size, shuffle=True)
    
    # 3. Training Loop Simulation
    print("Starting pipeline...")
    for step, (x_batch, y_batch) in enumerate(train_loader):
        # x_batch and y_batch are now jax.numpy arrays living on your accelerator (GPU/TPU)
        # You would pass them to your update step here:
        # state, loss = train_step(state, x_batch, y_batch)
        
        if step == 0:
            print(f"First batch images shape: {x_batch.shape}, dtype: {x_batch.dtype}")
            print(f"First batch labels shape: {y_batch.shape}, dtype: {y_batch.dtype}")
            break
