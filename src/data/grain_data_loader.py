import gzip
import os
import pathlib
from collections.abc import Iterator
from typing import NamedTuple

import grain.python as grain
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from loguru import logger

DATA_DIR = pathlib.Path(__file__).parent.resolve() / "store/mnist_data/"
DATA_DIR.mkdir(exist_ok=True, parents=True)

class LocalMNISTDataSource(grain.RandomAccessDataSource):
    """Idiomatic Grain Source for local OSSCI idx-ubyte.gz files."""
    def __init__(self, images_path: str, labels_path: str):
        
        # 1. Parse OSSCI custom idx image binary schema
        with gzip.open(images_path, 'rb') as f:
           
            # First 16 bytes contain metadata: magic number, sizes
            _ = np.frombuffer(f.read(4), dtype=np.dtype('>i4')) # Magic number
            num_images = int(np.frombuffer(f.read(4), dtype=np.dtype('>i4'))[0])
            rows = int(np.frombuffer(f.read(4), dtype=np.dtype('>i4'))[0])
            cols = int(np.frombuffer(f.read(4), dtype=np.dtype('>i4'))[0])
            

            # Read all pixels: shape (N, 28, 28)
            logger.info("Reading all image pixels...")
            self._images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        # 2. Parse OSSCI custom idx label binary schema
        with gzip.open(labels_path, 'rb') as f:
            # First 8 bytes contain metadata
            _ = np.frombuffer(f.read(4), dtype=np.dtype('>i4')) # Magic number
            num_labels = int(np.frombuffer(f.read(4), dtype=np.dtype('>i4'))[0])
            
            # Read all labels
            logger.info("Reaging all label pixels ")
            self._labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        assert len(self._images) == len(self._labels), "Images and labels count mismatch!"

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict:
        """Called by background workers. 

        Returns an individual dictionary element ready for JAX pipelines.
        """
        # Add a trailing channel dimension (28, 28, 1) standard for neural networks
        image = self._images[index][..., np.newaxis]
        label = self._labels[index]
        
        return {
            "image": image,
            "label": int(label)
        }

# 3. Define transformations (mapping logic)
def preprocess_mnist(element: dict) -> dict:
    # Convert uint8 (0-255) to normalized float32 (0.0-1.0)
    element["image"] = element["image"].astype(np.float32) / 255.0
    return element

# 4. Factory wrapper to build training/testing splits
def make_mnist_loader(images_file: str, labels_file: str, batch_size: int, is_train: bool):
    source = LocalMNISTDataSource(images_file, labels_file)
    
    # Wrap underlying source to Grain's optimization ecosystem
    dataset = grain.MapDataset.source(source)
    
    if is_train:
        # Globally shuffles array references seamlessly and deterministically
        dataset = dataset.shuffle(seed=42)
        
    dataset = dataset.map(preprocess_mnist)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=is_train)
    
    # Compile execution map down to a prefetched multi-threaded stream loop
    return dataset.to_iter_dataset()

def get_train_loader():
    return make_mnist_loader(
        images_file=os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"),
        labels_file=os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"),
        batch_size=64,
        is_train=True
    )

def get_test_loader():
    return make_mnist_loader(
        images_file=os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"),
        labels_file=os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"),
        batch_size=64,
        is_train=False,
    )


class ImageBatch(NamedTuple):
    images: Array  # Shape: (B, 1, H, W)
    labels: Array  # Shape: (B,)


def adapt_grain_batch(batch: dict) -> ImageBatch:
    """Converts a Grain dictionary batch into a PyTree-compatible ImageBatch.

    Transforms images from NHWC (B, H, W, 1) to NCHW (B, 1, H, W) as expected by Equinox Conv2d.
    """
    raw_images = jnp.asarray(batch["image"])  # Shape: (B, H, W, 1)
    raw_labels = jnp.asarray(batch["label"])  # Shape: (B,)

    # Permute from (B, H, W, C) to (B, C, H, W)
    nchw_images = jnp.transpose(raw_images, (0, 3, 1, 2))

    return ImageBatch(images=nchw_images, labels=raw_labels)


class GrainDatasetAdapter:
    """Wraps a Grain PyGrain dataset iterator to yield JAX ImageBatch tuples."""

    def __init__(self, grain_loader):
        self._grain_loader = grain_loader

    def __iter__(self) -> Iterator[ImageBatch]:
        for batch in self._grain_loader:
            yield adapt_grain_batch(batch)

def main():


    print(DATA_DIR)
    assert DATA_DIR.exists(), "given path does not exist"
    DATA_DIR = str(DATA_DIR)
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    # Test execution inside your training loop
    for batch in train_loader:
        x_batch = batch["image"]  # Pure NumPy Array
        y_batch = batch["label"]  # Pure NumPy Array
    
        print(f"Inputs Shape:  {x_batch.shape} (Type: {x_batch.dtype})")
        print(f"Targets Shape: {y_batch.shape} (Type: {y_batch.dtype})")
        break


if __name__ == "__main__":
    main()
