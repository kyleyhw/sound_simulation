from typing import Any, List

import numpy as np


def get_edge_indices(arr: np.ndarray) -> np.ndarray:
    """
    Gets the indices of elements on the edges of a NumPy ndarray.

    An element is on an edge if its index is 0 or -1 along any axis.

    Args:
        arr: An input NumPy ndarray of arbitrary dimensions.

    Returns:
        A 2D NumPy array of shape (N, arr.ndim) where each row is an
        edge index, and N is the number of edge elements.
    """
    # Create a boolean mask to mark the positions of edge elements.
    edge_mask = np.zeros(arr.shape, dtype=bool)

    # Iterate over each axis of the array.
    for axis in range(arr.ndim):
        # Mark the "start" edge of the current axis.
        # Mixed list of slice + int: declare as list[Any] so the int
        # assignment below does not trip the type-checker.
        start_slice: List[Any] = [slice(None)] * arr.ndim
        start_slice[axis] = 0
        edge_mask[tuple(start_slice)] = True

        # Mark the "end" edge, avoiding re-marking on axes of size 1.
        if arr.shape[axis] > 1:
            end_slice: List[Any] = [slice(None)] * arr.ndim
            end_slice[axis] = -1
            edge_mask[tuple(end_slice)] = True

    # Convert the boolean mask to a 2D array of indices.
    indices = np.stack(np.where(edge_mask), axis=-1)

    return indices


def set_edge_values(arr: np.ndarray, value) -> np.ndarray:
    """
    Sets the values of elements on the edges of a NumPy ndarray in-place.

    Args:
        arr (np.ndarray): An input NumPy ndarray to be modified.
        value: The new value to assign to all edge elements.

    Returns:
        np.ndarray: The modified input array with its edge values set to
                    the specified value.
    """
    # Get the indices of the edge elements.
    indices = get_edge_indices(arr)
    # Use the indices to set the value for all edge elements.
    arr[tuple(indices.T)] = value
    return arr
