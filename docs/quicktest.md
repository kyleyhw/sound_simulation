# Documentation for `quicktest.py`

## 1. Purpose

`quicktest.py` is a utility script designed to quickly generate a small set of dummy data files in the format expected by `datasetformat.py`. 

## 2. Rationale

The primary reason for this script is to facilitate testing of the data loading and training pipeline without needing to run a full, time-consuming simulation. By running this script, a developer can immediately create a valid dataset in the `data/train/` directory and test if the `datasetformat` class and the `trainingtest.py` script can load and process the data correctly.

## 3. Implementation Details

-   **Directory Creation**: It ensures the target directory `data/train/` exists.
-   **Data Generation**: It loops 16 times to create 16 dummy data files.
    -   **`audio`**: It generates random noise for the two audio channels. This is not physically realistic audio but has the correct shape `(2, T)`.
    -   **`speaker_xy`**: It uses a fixed, hard-coded set of speaker locations.
    -   **`room_rect`**: It uses a fixed, hard-coded room shape.
-   **Saving**: It saves the generated data into a compressed `.npz` file using `np.savez_compressed`, with the keys (`audio`, `speaker_xy`, `room_rect`) matching what the `datasetformat` class expects.
