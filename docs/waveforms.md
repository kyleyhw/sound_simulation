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

### `AudioFileWaveform`

Drives the simulation from a `.wav` file. Given a discrete audio signal
$u_k$ at native sample rate $f_a^{\text{native}}$ (Hz, as written by the
recording device), the constructor converts it to a simulation-time
sample rate

$$ f_a = f_a^{\text{native}} \cdot \tau, $$

where $\tau$ (`sim_time_per_second`, default 1.0) is the user's chosen
conversion factor between physical seconds and simulation time units.
For the dimensionless $c = 1, \Delta x = 1$ sandbox $\tau = 1$ keeps the
audio at its native speed; for a physical air simulation with
$c = 343 \text{ m/s}, \Delta x = 0.005 \text{ m}$ (where $\Delta t \approx 7 \mu\text{s}$
gives $f_s^{\text{sim}} \approx 137$ kHz), one would pass $\tau = 1$
again because both clocks are in seconds.

The source value at simulation time $t$ is computed by linear
interpolation between the two adjacent samples:

$$
p_{\text{src}}(t) = A \cdot \bigl[(1 - \alpha)\, u_k + \alpha\, u_{k+1}\bigr],
\quad k = \lfloor f_a (t - t_0) \rfloor, \quad \alpha = f_a (t - t_0) - k,
$$

for $t_0 \le t < t_0 + N / f_a$ (the audio's support window with delay
$t_0$ and length $N$). Outside that window $p_{\text{src}} = 0$.

**Aliasing constraint.** The simulation samples the source at rate
$1 / \Delta t$. Any audio content above $1 / (2 \Delta t)$ aliases. The
default 2D demo grid runs at $\Delta t = 0.5$, giving a sim-Nyquist of
1 cycle per simulation-time unit; an audio clip whose dominant
frequencies exceed that limit will be distorted. In a physical-units
simulation $\Delta t$ is typically $\sim 10^{-5}$ s so the sim-Nyquist
sits at $\sim 50$ kHz, comfortably above audible content.

**Sample-rate conversion.** Reading an int16 WAV normalises to float32
in $[-1, 1]$; stereo is converted to mono by averaging channels. The
raw float buffer is held on the private `_samples` attribute so
`data_io.SaveSimulationResults` skips it when dumping driver metadata
to HDF5 (HDF5 attrs cap at 64 KiB; even a one-second 16 kHz clip would
exceed that as an attr).

The default constructor (`path=""`) yields a silent source that returns
zero everywhere, which is what `build_waveform` in the backend falls
back to when a UI client sends an `AudioFileWaveform` spec without a
valid path. This keeps the engine demo-able even when the user has not
selected a file yet.

## 3. Design notes

- All waveforms inherit from `Waveform` and override `__call__(t)`. The
  callable interface lets `Driver.get_value(time)` be agnostic to the
  source family.
- `waveform_registry` maps string names to classes for reconstruction
  from JSON configuration sent by the frontend.
- Adding a new family is a one-class change: subclass `Waveform`,
  implement `__call__`, and register the name.
