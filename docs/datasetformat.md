# Documentation for `datasetformat.py`

## 1. Purpose

`datasetformat.py` defines a custom `Dataset` class for use with PyTorch. Its purpose is to bridge the gap between the raw simulation data saved on disk (in `.npz` format) and the format required by PyTorch's `DataLoader` for training the neural network.

## 2. Key Technologies

-   **PyTorch `Dataset`**: This script subclasses `torch.utils.data.Dataset`. This is the standard way to create a custom dataset in PyTorch. Any class that inherits from `Dataset` must implement two special methods:
    1.  `__len__(self)`: Returns the total number of samples in the dataset.
    2.  `__getitem__(self, i)`: Loads and returns the i-th sample from the dataset.
-   **NumPy `.npz` files**: The script assumes the data for each simulation scene is stored in a compressed NumPy `.npz` file.

## 3. Implementation Details

### Class: `datasetformat(Dataset)`

-   **`__init__(self, root)`**: The constructor finds all the `.npz` files in the specified `root` directory and stores their paths. This list of paths determines the size of the dataset.

-   **`__len__(self)`**: Simply returns the number of file paths found.

-   **`__getitem__(self, i)`**: This is the core logic of the dataset loader.
    1.  It takes an index `i`.
    2.  It gets the file path for that index: `self.paths[i]`.
    3.  It loads the `.npz` file using `np.load()`.
    4.  **Data Transformation**: It extracts the raw data from the file and transforms it into the specific input (`x`) and target (`y`) tensors that the model expects:
        -   `x`: The raw audio data is extracted and converted to a `float32` tensor.
        -   `y_spk`: The speaker coordinates `[[xL,yL],[xR,yR]]` are reshaped into a flat vector `[xL,yL,xR,yR]`.
        -   `y_room`: The room rectangle `[xmin, xmax, ymin, ymax]` is converted into a center-and-size format `[cx, cy, w, h]`. This is often a more stable representation for regression tasks in machine learning.
    5.  It converts the NumPy arrays to PyTorch tensors using `torch.from_numpy()` and returns the input tensor and the two target tensors.

### Design Rationale

-   **Lazy Loading**: The `__getitem__` method loads data one sample at a time, on-demand. This is a memory-efficient approach, as it means the entire dataset does not need to be loaded into RAM at once. This is crucial for handling large datasets.
-   **Decoupling**: This class decouples the data loading and preprocessing logic from the model training loop. The training loop can simply iterate over the `DataLoader`, which handles the complexities of fetching and batching the data in the background.
