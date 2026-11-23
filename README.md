# Aircraft Low-level-turbulence Simulator
## Idealised 3D Building-Wake Simulation

![Python](https://img.shields.io/badge/Language-Python-yellow?logo=python)

![SciPy](https://img.shields.io/badge/Library-SciPy-8CAAE6?logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-ffffff?logo=matplotlib&logoColor=black)

This is a simple Python conceptual model made for learning and visualisation for my application to HKO intern about wind shear and low-level-turbulence. It simulated low-level-turbulence (LLT) with a 3D constant wind field and how aircraft affected under it.


The project includes:

- Building-wake wind-field visualisation;
- A virtual aircraft flying through the wind field;
- A virtual Doppler-LIDAR RHI scan.
---

## Files

| File | Purpose |
|---|---|
| `wind_field.py` | Creates the idealised 3D wind field and saves \(x\)-\(y\) and \(x\)-\(z\) plots/animations |
| `aircraft.py` | Simulates a virtual aircraft moving through the wind field |
| `virtual_lidar.py` | Creates synthetic RHI Doppler-LIDAR radial-velocity plots |

---

## Wind-field model

The model uses a background wind:

$$
\mathbf{V}_0 = (U_0, 0, 0),
\qquad U_0 = 10\ \mathrm{m\,s^{-1}}.
$$

A rectangular building is placed in the domain:

$$
30\ \mathrm{m} \times 30\ \mathrm{m} \times 100\ \mathrm{m}.
$$

The wake includes:

- Reduced wind speed behind the building;
- Oscillating crosswind disturbance;
- Upward motion near the roof;
- Downward motion behind the building;
- Wake spreading and decay downstream.

The wind field is:

$$
\mathbf{V}(x,y,z,t)=(u,v,w).
$$

---

## Wind-shear diagnostic

The vertical wind-shear magnitude is calculated as:

$$
S_z =
\sqrt{
\left(\frac{\partial u}{\partial z}\right)^2 +
\left(\frac{\partial v}{\partial z}\right)^2 +
\left(\frac{\partial w}{\partial z}\right)^2
}.
$$
---

## Virtual RHI-LIDAR

The virtual LIDAR performs an RHI scan in the vertical \(x\)-\(z\) plane.

For a beam with elevation angle $\theta$, the synthetic Doppler radial velocity is:

$$
V_r = u\cos\theta + w\sin\theta.
$$

The LIDAR measures radial velocity along the beam, not the full wind vector.

---

## Installation

```bash
pip install numpy matplotlib scipy pillow
```

## Usage

Generate wind-field plots and animations:

```bash
python wind_field.py
```

Run the virtual aircraft experiment:

```bash
python aircraft.py
```

Run the virtual RHI-LIDAR experiment:

```bash
python virtual_lidar.py
```

---

## Main outputs

- `xy_flow_animation.gif`
- `xz_flow_animation.gif`
- `xy_flow_t10s.png`, `xy_flow_t30s.png`, etc.
- `xz_flow_t10s.png`, `xz_flow_t30s.png`, etc.
- `aircraft_ground_track.png`
- `aircraft_velocity_timeseries.png`
- `aircraft_crosswake_animation.gif`
- `synthetic_rhi_t10s.png`, `synthetic_rhi_t30s.png`, etc.

---

## Limitations

- The wake is prescribed analytically.
- No Navier–Stokes or CFD solver is used.
- No pressure, temperature, stability, or turbulence-closure model is included.
- The aircraft model is kinematic only.
- The LIDAR is idealised and contains no noise, attenuation, or missing data.
