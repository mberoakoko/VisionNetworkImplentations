import gzip
import os
import pathlib

import grain.python as grain
import numpy as np
from loguru import logger

DATA_DIR = ( pathlib.Path(__file__).parent ).resolve()  / "store/mnist_data/"

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
            self._images = np.frombuffer(f.read(), dicttype=np.uint8).reshape(num_images, rows, cols)

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

def main():

    DATA_DIR = ( pathlib.Path(__file__).parent ).resolve()  / "store/mnist_data/"
    print(DATA_DIR)
    assert DATA_DIR.exists(), "given path does not exist"
    DATA_DIR = str(DATA_DIR)
    train_loader = make_mnist_loader(
        images_file=os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"),
        labels_file=os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"),
        batch_size=64,
        is_train=True
    )

    test_loader = make_mnist_loader(
        images_file=os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"),
        labels_file=os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"),
        batch_size=64,
        is_train=False
    )

    # Test execution inside your training loop
    for batch in train_loader:
        x_batch = batch["image"]  # Pure NumPy Array
        y_batch = batch["label"]  # Pure NumPy Array
    
        print(f"Inputs Shape:  {x_batch.shape} (Type: {x_batch.dtype})")
        print(f"Targets Shape: {y_batch.shape} (Type: {y_batch.dtype})")
        break


if __name__ == "__main__":
    main()
