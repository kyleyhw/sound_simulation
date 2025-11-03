# Documentation for `cnn1d.py`

## 1. Purpose

`cnn1d.py` defines the core machine learning model for this project: a 1D Convolutional Neural Network (CNN). This network is designed to take the raw audio time-series data from the sensors as input and predict the physical properties of the environment, specifically the room shape and the location of the sound sources.

## 2. Scientific Principles

-   **Convolutional Neural Networks (CNNs)**: CNNs are a class of deep learning models particularly well-suited for processing grid-like data, such as images (2D) or time-series (1D). They use convolutional layers to apply filters (kernels) across the input data, which allows them to learn hierarchical patterns.
    -   **1D Convolutions**: In this model, 1D convolutions are used to slide filters along the time axis of the audio data. This enables the network to learn to recognize temporal patterns, such as the timing and shape of echoes, which are crucial for inferring spatial information.

-   **Multi-Task Learning**: The model has a shared "backbone" and two separate "heads." This is a form of multi-task learning where the network learns a shared representation of the input data (from the backbone) and then uses that representation to make two different predictions (room shape and speaker location). This can be more efficient than training two separate models.

## 3. Implementation Details

### Class: `cnn1d(nn.Module)`

This class defines the neural network architecture using PyTorch's `nn.Module`.

-   **`__init__(...)`**: The constructor defines the layers of the network.
    -   **`self.backbone`**: This is a sequence of `Conv1d`, `ReLU` activation, and `MaxPool1d` layers.
        -   `nn.Conv1d`: Learns to extract temporal features from the audio.
        -   `nn.ReLU`: A standard activation function that introduces non-linearity.
        -   `nn.MaxPool1d`: Downsamples the data along the time axis, making the representation more compact and allowing subsequent layers to have a larger receptive field.
        -   `nn.AdaptiveAvgPool1d(1)`: This powerful layer averages over the entire time dimension, producing a fixed-size vector (of size 128) that summarizes the temporal features extracted by the convolutional layers. This allows the network to handle inputs of varying time lengths.

    -   **`self.head_room`**: A sequence of fully-connected (`nn.Linear`) layers that takes the 128-feature vector from the backbone and predicts 4 numbers representing the room shape (center x, center y, width, height).

    -   **`self.head_spk`**: A similar head that predicts 4 numbers representing the locations of two speakers (xL, yL, xR, yR).

-   **`forward(self, x)`**: This method defines the forward pass of the network.
    -   It takes a batch of audio data `x` of shape `[Batch, Channels, Time]`.
    -   It passes the data through the `backbone` to get the shared feature representation `h`.
    -   It then passes `h` through both heads to get the final predictions for the room and speakers.
    -   It returns the predictions in a dictionary, which is a clean way to handle multiple outputs.
