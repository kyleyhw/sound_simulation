# Documentation for `trainingtest.py`

## 1. Purpose

`trainingtest.py` is the main script for training the 1D CNN model defined in `cnn1d.py`. It handles setting up the dataset, initializing the model and optimizer, and running the training loop for a fixed number of epochs.

## 2. Implementation Details

### `main()` Function

-   **Device Selection**: It first determines the best available hardware for training (NVIDIA GPU with `cuda`, Apple Silicon with `mps`, or `cpu`) to ensure the model runs on the fastest possible device.

-   **Dataset and DataLoader**: 
    -   It instantiates the `datasetformat` class to point to the training data located in `data/train/`.
    -   It wraps the dataset in a `torch.utils.data.DataLoader`. The `DataLoader` is a powerful PyTorch utility that automatically handles:
        -   **Batching**: Grouping individual samples into batches.
        -   **Shuffling**: Randomizing the order of data at every epoch to prevent the model from learning the order of the training set.
        -   **Parallelization** (optional, via `num_workers`): Loading data in parallel on multiple CPU cores.

-   **Model, Optimizer, and Loss Function**:
    -   **Model**: It creates an instance of the `cnn1d` model and moves it to the selected device.
    -   **Optimizer**: It uses `torch.optim.AdamW`, a sophisticated and widely used optimization algorithm that adjusts the learning rate for each parameter individually.
    -   **Loss Function**: It uses `nn.SmoothL1Loss`. This loss function is a variation of Mean Absolute Error (L1 Loss) that behaves like Mean Squared Error (L2 Loss) when the error is small. It is often used in regression tasks as it is less sensitive to outliers than MSE.

-   **Training Loop**:
    -   The script iterates for a fixed number of epochs (20 in this case).
    -   Within each epoch, it iterates through the batches provided by the `DataLoader`.
    -   **For each batch**:
        1.  **Data Preprocessing**: It normalizes the input audio `x` by subtracting the mean and dividing by the standard deviation. This is a standard practice that helps stabilize training.
        2.  **Forward Pass**: It passes the input `x` through the network to get the model's predictions (`out`).
        3.  **Loss Calculation**: It calculates the loss by comparing the model's predictions (`out["spk"]`, `out["room"]`) with the ground truth labels (`y_spk`, `y_room`). The room loss is weighted by 0.5.
        4.  **Backward Pass & Optimization**: 
            -   `opt.zero_grad()`: Clears old gradients.
            -   `loss.backward()`: Computes the gradient of the loss with respect to all model parameters (backpropagation).
            -   `opt.step()`: Updates the model's parameters based on the computed gradients.
    -   At the end of each epoch, it prints the average loss for that epoch.

## 3. Design Rationale

-   **Standard Training Structure**: This script follows a very standard and well-established PyTorch training pipeline, making it easy to understand for anyone familiar with the framework.
-   **Clear Separation of Concerns**: It clearly separates the concerns of data loading (`datasetformat`, `DataLoader`), model architecture (`cnn1d`), and the training process itself.
