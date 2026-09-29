import gzip
import os
import pathlib
import typing
import urllib.request as lib_request
from urllib.error import HTTPError, URLError

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger


class FileProxy_t(typing.TypedDict):
    train_images: str 
    train_labels: str 
    test_images: str 
    test_labels: str  

type TrainingTestData_t = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _download_progress(block_count, block_size, total_size):
    """Callback function to track download progress."""
    if total_size > 0:
        # Calculate how many bytes have been downloaded
        downloaded = block_count * block_size
        # Calculate percentage
        percentage = (downloaded / total_size) * 100
        # Limit to 100% (sometimes the last block slightly exceeds total_size)
        percentage = min(percentage, 100)
        print(f"\033[1;32mDownloaded: {percentage:.2f}%\033[0m", end="\r", flush=True)
    else:
        print("\033[1;32mDownloading... (Unknown total size)\033[0m", end="\r")



class MNIST_Loader: 
    _URL_BASE: str = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    def __init__(self, data_dir: str = r"data/store"):
        
        self._data_dir = str ( 
            (pathlib.Path(__file__).parent.parent / data_dir)
            .resolve()
        )
        os.makedirs(self._data_dir, exist_ok=True)
        logger.info(f"[{__class__.__name__}]::Selected cache path {self._data_dir}")
        
        self._FILES: FileProxy_t = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz"
        }

        self.train_images, self.train_labels, self.test_images, self.test_labels = self._load_data()
    

    def _download_data_if_needed(self, file_name: str ) -> str:

        target_filepath: str = str ( pathlib.Path(__file__).parent / self._data_dir / file_name )
        logger.info(f"[{__class__.__name__}]::Target FilePath {target_filepath}")

        if not os.path.exists(target_filepath):
            logger.info(f"[ Downloder ] Downloading {file_name}...")
            try:
                url:str = self._URL_BASE + file_name 
                logger.info(f"Targeting url {url}")
                lib_request.urlretrieve(
                    url,
                    filename=target_filepath,
                    reporthook=_download_progress 
                )

                logger.info(f"[ Downloader ] Successfully Downloaded::{target_filepath} ")
                return target_filepath 

            except HTTPError as e:
                logger.error(f"HTTP_ERROR occurred {e} : {e.getcode()} | {e.info}")
                raise e 
            
            except URLError as e:
                logger.error(f"URL_ERROR occued {e}")

            except Exception as e:
                logger.error(f"An expected error occured {e}")
                raise e 
        logger.info(f"Resources found in disk... loading from disk::{target_filepath}")
        return target_filepath

    def _load_data(self) -> TrainingTestData_t: 
        def load_images(filename: str) -> np.ndarray:
            path = self._download_data_if_needed(filename)
            try:
                with gzip.open(path, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=16)
                if data.size % 784 != 0 or data.size == 0:
                    logger.error(f"Invalid byte cound {data.size} for {filename}")
                    raise ValueError(f"Invalid byte count {data.size} for {filename}")
                return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0
            except ValueError:

                logger.warning("Purging corrupted file, reattempting download...")
                
                if os.path.exists(path):
                    logger.warning("Removing existing file...")
                    os.remove(path)
                
                path = self._download_data_if_needed(filename)
                
                with gzip.open(path, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8, offset=16)
                    logger.debug(f"[ Load Images  ]: {data.shape=}")
                return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

        def load_labels(filename: str) -> np.ndarray:
            path = self._download_data_if_needed(filename)
            try:
                with gzip.open(path, 'rb') as f:
                    return np.frombuffer(f.read(), np.uint8, offset=8)
            except Exception:
                logger.warning("Purging existing label file... ")
                if os.path.exists(path):
                    os.remove(path)
                path = self._download_data_if_needed(filename)
                with gzip.open(path, 'rb') as f:
                    return np.frombuffer(f.read(), np.uint8, offset=8)


        return (
            load_images(self._FILES["train_images"]),
            load_labels(self._FILES["train_labels"]),
            load_images(self._FILES["test_images"]),
            load_labels(self._FILES["test_labels"])
        )

    def get_batches(self, rng_key: jax.Array, batch_size: int, split: str = "train"):
        """Yield mini-batches as pure JAX arrays."""
        if split == "train":
            x, y = self.train_images, self.train_labels
            # Shuffle indices using JAX random key
            indices = jax.random.permutation(rng_key, len(x))
            indices_np = np.array(indices)
            x, y = x[indices_np], y[indices_np]
        else:
            x, y = self.test_images, self.test_labels

        num_batches = len(x) // batch_size
        for i in range(num_batches):
            bx = jnp.array(x[i * batch_size : (i + 1) * batch_size])
            by = jnp.array(y[i * batch_size : (i + 1) * batch_size])
            yield bx, by
