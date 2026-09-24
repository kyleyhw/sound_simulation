# `utils.py` — edge helpers

- `get_edge_indices(arr)`: the coordinates of every element on any face
  of an N-dimensional array. It builds a boolean mask with index 0 and
  index −1 set along each axis, then applies `np.where`.
- `set_edge_values(arr, value)`: assigns `value` to every face element.
  The 1D / N-D fallback path of `Simulate.step()` uses it to enforce the
  $p = 0$ outer boundary. The fused 2D and 3D kernels zero their faces
  inline.

`get_edge_values` and the unseeded `LocationGenerator` were removed in
the Phase 3 audit because nothing used them.
