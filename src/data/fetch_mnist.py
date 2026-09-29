import pathlib
import sys
import urllib.request

# Fill this in with your desired destination path
_ROOT_DATA_FOLDER: pathlib.Path = pathlib.Path(__file__).parent / "store/mnist_data" 

# Using the reliable OSSCI PyTorch mirror, as Yann LeCun's original site 
# often throws 403 Forbidden or rate-limits automated scripts nowadays.
_MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

_MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

def _download_with_progress(url: str, dest_path: pathlib.Path) -> None:
    """Downloads a file with a CLI progress bar, skipping if it already exists."""
    # Pre-check and short circuit to minimize server calls
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[{dest_path.name}] already exists. Skipping download.")
        return

    # Ensure the target directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        
        # Format the progress string
        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            progress_str = f"Downloading {dest_path.name}: {percent:3d}% | {mb_downloaded:.2f} MB / {mb_total:.2f} MB"
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            progress_str = f"Downloading {dest_path.name}: {mb_downloaded:.2f} MB (total size unknown)"
        
        # The carriage return '\r' sends the cursor back to the start of the line
        # Pad with spaces to overwrite any longer string that was previously there
        sys.stdout.write(f"\r{progress_str:<80}")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=_reporthook)
        print()  # Move to the next line once 100% is reached
    except Exception as e:
        # Clean up corrupted/partial files if the download fails or is interrupted
        print(f"\nError downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def download_mnist(target_dir: pathlib.Path = _ROOT_DATA_FOLDER) -> None:
    """
    Downloads the 4 core MNIST gzip files to the target directory.
    Safe to call multiple times; cached files will be skipped.
    """
    if str(target_dir) == "...":
        raise ValueError("Please update _ROOT_DATA_FOLDER with a valid path.")
        
    print(f"Checking MNIST dataset in: {target_dir.absolute()}")
    
    for filename in _MNIST_FILES:
        url = _MNIST_BASE_URL + filename
        file_path = target_dir / filename
        _download_with_progress(url, file_path)
        
    print("MNIST dataset is ready.")

if __name__ == "__main__":
    # Example usage:
    # _ROOT_DATA_FOLDER = pathlib.Path("./mnist_data")
    download_mnist()
