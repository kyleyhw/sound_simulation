import numpy as np
from typing import List, Tuple

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
        start_slice = [slice(None)] * arr.ndim
        start_slice[axis] = 0
        edge_mask[tuple(start_slice)] = True

        # Mark the "end" edge, avoiding re-marking on axes of size 1.
        if arr.shape[axis] > 1:
            end_slice = [slice(None)] * arr.ndim
            end_slice[axis] = -1
            edge_mask[tuple(end_slice)] = True

    # Convert the boolean mask to a 2D array of indices.
    indices = np.stack(np.where(edge_mask), axis=-1)

    return indices


def get_edge_values(arr: np.ndarray) -> np.ndarray:
    """
    Gets the values of elements on the edges of a NumPy ndarray.

    Args:
        arr: An input NumPy ndarray.

    Returns:
        A 1D NumPy array containing the values of the edge elements.
    """
    # Get the indices of the edge elements.
    indices = get_edge_indices(arr)
    # NumPy's advanced indexing requires indices to be a tuple of arrays.
    # We transpose the (N, D) array of indices to a (D, N) tuple of arrays.
    return arr[tuple(indices.T)]

def set_edge_values(arr: np.ndarray, value) -> np.ndarray:
    """
    Sets the values of elements on the edges of a NumPy ndarray in-place.

    Args:
        arr: An input NumPy ndarray to be modified.
        value: The new value to assign to all edge elements.
    """
    # Get the indices of the edge elements.
    indices = get_edge_indices(arr)
    # Use the indices to set the value for all edge elements.
    arr[tuple(indices.T)] = value
    return arr

if __name__ == '__main__':
    print("--- Array Demonstration ---")
    test_array = np.array([
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21]
    ])

    print("Original Array:")
    print(test_array)
    print(f"\nShape: {test_array.shape}\n")

    # --- Using get_edge_indices ---
    edge_indices = get_edge_indices(test_array)
    print("Edge Indices:")
    print(edge_indices)

    # --- Using get_edge_values ---
    print("\n" + "=" * 40 + "\n")
    edge_values = get_edge_values(test_array)
    print("Edge Values:")
    print(edge_values)

    # --- Using set_edge_values ---
    print("\n" + "=" * 40 + "\n")
    print("Setting edge values to 99...")
    # Create a copy to avoid modifying the original array used in other examples
    modified_array = test_array.copy()
    set_edge_values(modified_array, 99)

    print("\nArray after setting edge values:")
    print(modified_array)