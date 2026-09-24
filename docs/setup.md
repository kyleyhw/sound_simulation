# `setup.py` — `Driver` and `Sensor`

## `Driver` (frozen dataclass)

A point source: a grid `position` and a callable `waveform`
$p_\mathrm{src}(t)$. `get_value(t)` returns `float(waveform(t))`.

- **Frozen.** `Simulate` caches the single-driver fast path, i.e. the
  index of the only driver. If a driver's position could be changed in
  place, that cache would silently go stale and the source would keep
  emitting at the old cell. To move a source, build a new `Driver` and
  call `Simulate.set_drivers` or `add_driver`.
- **Normalised position.** `__post_init__` converts the position to a
  tuple of Python `int`s, rounding float coordinates. A float index would
  otherwise fail inside NumPy on the first step.
- **Injection.** The value evaluated at $t_n$ is *added* to $p^{n+1}$
  after wall and obstacle zeroing (a soft source). See `simulate.md`.

## `Sensor` (dataclass)

A recording point with `position`, and `timeseries` / `sample_rate`
fields that the caller fills. `dataset.run_with_sensors` records
sensors in streaming fashion and fills both fields. `Simulate.step()`
does not sample sensors itself.

## Removed

The random factories `GenerateDriver` and `GenerateSensor`, together
with `utils.LocationGenerator`, were used only by the deleted legacy
`generate.py`. `LocationGenerator` drew from the unseeded global
`np.random`. Seeded placement lives in `dataset.random_free_position`
and `dataset.pick_mic_positions`.
