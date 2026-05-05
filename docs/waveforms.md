# Documentation for `waveforms.py`

## 1. Purpose

Defines callable `Waveform` objects $p_{\text{src}}(t)$ that drive
point-source injection in the FDTD engine. A `Driver` holds a grid
position and one of these waveforms; on every step the engine evaluates
`waveform(t)` and adds the value to the pressure field.

## 2. Available waveforms

### `Cosine`

A continuous monochromatic source,

$$ p_{\text{src}}(t) = A \cos(2 \pi f t). $$

For a leap-frog FDTD with timestep $\Delta t$, the sample-rate Nyquist
constraint is $f \Delta t < 0.5$; ten or more samples per period
($f \Delta t \le 0.1$) is recommended for visually clean propagation.
Choosing $f \Delta t \ge 0.5$ silently aliases or — at the exact
half-rate — collapses the source to a constant DC injection that
monotonically pumps energy into the field. This was the dominant bug in
the initial web UI demo and is the reason the default UI source was
switched to a Ricker wavelet.

### `GaussianPulse`

$$ p_{\text{src}}(t) = A \exp\!\left( -\frac{(t - t_0)^2}{2 \sigma^2} \right). $$

Single-lobe transient, low-pass and DC-rich. Useful for impulse-response
work but can leave a quasi-static residual in closed hard-walled
domains.

### `RickerWavelet`

The Ricker (Mexican-hat) wavelet — the second derivative of a Gaussian:

$$ p_{\text{src}}(t) = A \, \big( 1 - 2 (\pi f (t - t_0))^2 \big) \exp\!\left( -(\pi f (t - t_0))^2 \right). $$

Three properties make it the standard FDTD excitation:

1. **Mean-zero in the limit.** The integral over $\mathbb{R}$ is zero,
   so the source deposits no net mass / DC into the field.
2. **Dominant-frequency control.** The amplitude spectrum peaks at $f$,
   making it easy to band-limit the simulation against the grid's
   numerical dispersion budget.
3. **Compact temporal support.** Negligible amplitude outside
   $|t - t_0| \gtrsim 2/f$, so taking $t_0 \ge 1/f$ guarantees the
   pulse starts and ends near zero.

Defaults (`amplitude=1`, `frequency=0.1`, `delay=20.0`) are tuned for
the demo grid: dominant period $\approx 10$ steps at $\Delta t = 0.5$,
pulse width $\approx 4/f = 40$ time units centred at $t = 20$.

## 3. Design notes

- All waveforms inherit from `Waveform` and override `__call__(t)`. The
  callable interface lets `Driver.get_value(time)` be agnostic to the
  source family.
- `waveform_registry` maps string names to classes for reconstruction
  from JSON configuration sent by the frontend.
- Adding a new family is a one-class change: subclass `Waveform`,
  implement `__call__`, and register the name.
