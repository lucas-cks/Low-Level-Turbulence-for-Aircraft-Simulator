import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from wind_field import (
    x, y, z,
    X, Y, Z,
    U0,
    building_width,
    building_length,
    building_height,
    building_mask,
    generate_flow
)

# Virtual runway and aircraft configuration

# The runway centreline is a line parallel to the y direction.
# Cross-wake traversal experiment
runway_x = 100.0
runway_y_start = 120.0
runway_y_end = -120.0

# Aircraft starts at y = +120 m and flies toward y = -120 m.
x_start = runway_x
y_start = runway_y_start
z_start = 60.0

# Constant commanded air-relative velocity.
# The aircraft intends to fly in the negative-y direction.
commanded_air_velocity = np.array([
    0.0,     # x direction
    -120.0,  # y direction
    0.0      # z direction
])

# The aircraft travels 240 m in 2 s without wind.
t_start = 10.0
flight_duration = 2.0
dt_aircraft = 0.02

time = np.arange(
    t_start,
    t_start + flight_duration + dt_aircraft,
    dt_aircraft
)

time_elapsed = time - t_start
n_steps = len(time)

# sample wind at one aircraft position
def wind_at_point(x_pos, y_pos, z_pos, t):

    # generate_flow() can operate on 1x1x1 arrays.
    x_point = np.array([[[x_pos]]])
    y_point = np.array([[[y_pos]]])
    z_point = np.array([[[z_pos]]])

    u_point, v_point, w_point = generate_flow(
        x_point,
        y_point,
        z_point,
        t=t,
        U0=U0
    )

    return np.array([
        float(u_point.squeeze()),
        float(v_point.squeeze()),
        float(w_point.squeeze())
    ])

# Simulate the aircraft ground track
position = np.zeros((n_steps, 3))
wind_velocity = np.zeros((n_steps, 3))
ground_velocity = np.zeros((n_steps, 3))

position[0] = np.array([
    x_start,
    y_start,
    z_start
])

for i in range(n_steps - 1):

    # Wind at the aircraft's current location and current time
    wind_velocity[i] = wind_at_point(
        position[i, 0],
        position[i, 1],
        position[i, 2],
        time[i]
    )

    # Simplified kinematic ground velocity:
    # aircraft commanded air velocity + local wind velocity
    ground_velocity[i] = (
        commanded_air_velocity
        + wind_velocity[i]
    )

    # First-order position update
    position[i + 1] = (
        position[i]
        + ground_velocity[i] * dt_aircraft
    )

# Calculate the final point
wind_velocity[-1] = wind_at_point(
    position[-1, 0],
    position[-1, 1],
    position[-1, 2],
    time[-1]
)

ground_velocity[-1] = (
    commanded_air_velocity
    + wind_velocity[-1]
)

# Nominal no-wind trajectory
nominal_x = np.full_like(
    time_elapsed,
    x_start
)

nominal_y = y_start + (
    commanded_air_velocity[1] * time_elapsed
)

nominal_z = np.full_like(
    time_elapsed,
    z_start
)

# Key quantities along the aircraft trajectory

# Ground-velocity components
ground_vx = ground_velocity[:, 0]
ground_vy = ground_velocity[:, 1]
ground_vz = ground_velocity[:, 2]

# Wind-induced perturbations relative to intended airspeed
delta_vx = ground_vx - commanded_air_velocity[0]
delta_vy = ground_vy - commanded_air_velocity[1]
delta_vz = ground_vz - commanded_air_velocity[2]

# Cross-track deviation from the intended runway centreline x = 100 m
cross_track_deviation = position[:, 0] - runway_x

# Along-runway deviation relative to the nominal position
along_track_deviation = position[:, 1] - nominal_y

# Vertical deviation from the nominal constant-altitude path
vertical_deviation = position[:, 2] - nominal_z

# Static x-y ground-track plot
z_plot = z_start
z_index = np.argmin(np.abs(z - z_plot))

u_field, v_field, w_field = generate_flow(
    X,
    Y,
    Z,
    t=t_start,
    U0=U0
)

speed_field = np.sqrt(
    u_field**2
    + v_field**2
    + w_field**2
)

speed_xy = speed_field[:, :, z_index]
u_xy = u_field[:, :, z_index]
v_xy = v_field[:, :, z_index]

building_xy = building_mask[:, :, z_index]

speed_xy = np.where(
    building_xy,
    np.nan,
    speed_xy
)

u_xy = np.where(
    building_xy,
    np.nan,
    u_xy
)

v_xy = np.where(
    building_xy,
    np.nan,
    v_xy
)

fig_track, ax_track = plt.subplots(figsize=(12, 7))

mesh = ax_track.pcolormesh(
    X[:, :, z_index],
    Y[:, :, z_index],
    speed_xy,
    shading="auto",
    cmap="viridis",
    vmin=0,
    vmax=1.3 * U0
)

ax_track.quiver(
    X[::5, ::5, z_index],
    Y[::5, ::5, z_index],
    u_xy[::5, ::5],
    v_xy[::5, ::5],
    color="white",
    scale=180,
    width=0.002
)

# Building footprint
building_x = [
    0,
    building_length,
    building_length,
    0,
    0
]

building_y = [
    -building_width / 2,
    -building_width / 2,
    building_width / 2,
    building_width / 2,
    -building_width / 2
]

ax_track.plot(
    building_x,
    building_y,
    color="red",
    linewidth=3,
    label="Building footprint"
)

# Intended straight runway/flight track
ax_track.plot(
    nominal_x,
    nominal_y,
    linestyle="--",
    color="white",
    linewidth=2,
    label="Nominal no-wind trajectory"
)

# Simulated wind-perturbed ground track
ax_track.plot(
    position[:, 0],
    position[:, 1],
    color="yellow",
    linewidth=3,
    label="Wind-perturbed ground track"
)

ax_track.scatter(
    position[0, 0],
    position[0, 1],
    color="lime",
    s=100,
    zorder=5,
    label="Aircraft start"
)

ax_track.scatter(
    position[-1, 0],
    position[-1, 1],
    color="red",
    s=100,
    zorder=5,
    label="Aircraft end"
)

ax_track.set_xlim(x.min(), x.max())
ax_track.set_ylim(y.min(), y.max())

ax_track.set_xlabel("x direction (m)")
ax_track.set_ylabel("y direction / virtual runway direction (m)")

ax_track.set_title(
    "Virtual aircraft cross-wake trajectory at z = 60 m"
)

ax_track.legend(loc="upper right")

fig_track.colorbar(
    mesh,
    ax=ax_track,
    label="Wind speed magnitude (m s$^{-1}$)"
)

fig_track.tight_layout()
fig_track.savefig(
    "aircraft_ground_track.png",
    dpi=300
)

plt.show()

# Aircraft velocity and perturbation time series

fig_velocity, axes = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    sharex=True
)

# Full ground-velocity components
axes[0].plot(
    time_elapsed,
    ground_vx,
    linewidth=2,
    label=r"Ground velocity $V_{g,x}$"
)

axes[0].plot(
    time_elapsed,
    ground_vy,
    linewidth=2,
    label=r"Ground velocity $V_{g,y}$"
)

axes[0].axhline(
    commanded_air_velocity[0],
    color="black",
    linestyle="--",
    alpha=0.7
)

axes[0].axhline(
    commanded_air_velocity[1],
    color="black",
    linestyle="--",
    alpha=0.7,
    label="Commanded air-relative velocity"
)

axes[0].set_ylabel("Ground velocity (m s$^{-1}$)")
axes[0].set_title(
    "Aircraft ground-velocity components"
)

axes[0].grid(alpha=0.3)
axes[0].legend()

# Wind-induced perturbations
axes[1].plot(
    time_elapsed,
    delta_vx,
    linewidth=2,
    label=r"$\Delta V_x = u$"
)

axes[1].plot(
    time_elapsed,
    delta_vy,
    linewidth=2,
    label=r"$\Delta V_y = v$"
)

axes[1].plot(
    time_elapsed,
    delta_vz,
    linewidth=2,
    label=r"$\Delta V_z = w$"
)

axes[1].axhline(
    0,
    color="black",
    linewidth=1
)

axes[1].set_xlabel("Time since aircraft entry (s)")
axes[1].set_ylabel("Wind-induced velocity perturbation (m s$^{-1}$)")
axes[1].set_title(
    "Local wind components encountered by virtual aircraft"
)

axes[1].grid(alpha=0.3)
axes[1].legend()

fig_velocity.tight_layout()
fig_velocity.savefig(
    "aircraft_velocity_timeseries.png",
    dpi=300
)

plt.show()

# Track-deviation plot

fig_deviation, axes = plt.subplots(
    3,
    1,
    figsize=(11, 9),
    sharex=True
)

axes[0].plot(
    time_elapsed,
    cross_track_deviation,
    color="tab:red",
    linewidth=2
)

axes[0].axhline(
    0,
    color="black",
    linewidth=1
)

axes[0].set_ylabel(r"$x-x_0$ (m)")
axes[0].set_title("Cross-track deviation from nominal runway centreline")
axes[0].grid(alpha=0.3)

axes[1].plot(
    time_elapsed,
    along_track_deviation,
    color="tab:blue",
    linewidth=2
)

axes[1].axhline(
    0,
    color="black",
    linewidth=1
)

axes[1].set_ylabel(r"$y-y_{\mathrm{nominal}}$ (m)")
axes[1].set_title("Along-track deviation from nominal position")
axes[1].grid(alpha=0.3)

axes[2].plot(
    time_elapsed,
    vertical_deviation,
    color="tab:green",
    linewidth=2
)

axes[2].axhline(
    0,
    color="black",
    linewidth=1
)

axes[2].set_xlabel("Time since aircraft entry (s)")
axes[2].set_ylabel(r"$z-z_0$ (m)")
axes[2].set_title("Vertical displacement caused by idealised vertical wind")
axes[2].grid(alpha=0.3)

fig_deviation.tight_layout()
fig_deviation.savefig(
    "aircraft_track_deviations.png",
    dpi=300
)

plt.show()

# Animation: aircraft moving through the wake

fig_anim, ax_anim = plt.subplots(figsize=(12, 7))

def update(frame):

    ax_anim.clear()

    current_time = time[frame]

    u_frame, v_frame, w_frame = generate_flow(
        X,
        Y,
        Z,
        t=current_time,
        U0=U0
    )

    speed_frame = np.sqrt(
        u_frame**2
        + v_frame**2
        + w_frame**2
    )

    speed_xy_frame = speed_frame[:, :, z_index]
    u_xy_frame = u_frame[:, :, z_index]
    v_xy_frame = v_frame[:, :, z_index]

    speed_xy_frame = np.where(
        building_xy,
        np.nan,
        speed_xy_frame
    )

    u_xy_frame = np.where(
        building_xy,
        np.nan,
        u_xy_frame
    )

    v_xy_frame = np.where(
        building_xy,
        np.nan,
        v_xy_frame
    )

    ax_anim.pcolormesh(
        X[:, :, z_index],
        Y[:, :, z_index],
        speed_xy_frame,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1.3 * U0
    )

    ax_anim.quiver(
        X[::5, ::5, z_index],
        Y[::5, ::5, z_index],
        u_xy_frame[::5, ::5],
        v_xy_frame[::5, ::5],
        color="white",
        scale=180,
        width=0.002
    )

    ax_anim.plot(
        building_x,
        building_y,
        color="red",
        linewidth=3,
        label="Building footprint"
    )

    ax_anim.plot(
        nominal_x,
        nominal_y,
        linestyle="--",
        color="white",
        linewidth=1.5,
        label="Nominal trajectory"
    )

    ax_anim.plot(
        position[:frame + 1, 0],
        position[:frame + 1, 1],
        color="yellow",
        linewidth=2.5,
        label="Actual ground track"
    )

    ax_anim.scatter(
        position[frame, 0],
        position[frame, 1],
        color="cyan",
        edgecolor="black",
        s=130,
        marker=">",
        zorder=5,
        label="Virtual aircraft"
    )

    ax_anim.set_xlim(x.min(), x.max())
    ax_anim.set_ylim(y.min(), y.max())

    ax_anim.set_xlabel("x direction (m)")
    ax_anim.set_ylabel("y direction / virtual runway direction (m)")

    ax_anim.set_title(
        f"Virtual aircraft crossing idealised wake: "
        f"t = {current_time:.2f} s"
    )

    ax_anim.legend(loc="upper right")

    return []

animation = FuncAnimation(
    fig_anim,
    update,
    frames=n_steps,
    interval=50,
    blit=False
)

animation.save(
    "aircraft_crosswake_animation.gif",
    writer="pillow",
    fps=20
)

plt.close(fig_anim)

print("Saved:")
print("- aircraft_ground_track.png")
print("- aircraft_velocity_timeseries.png")
print("- aircraft_track_deviations.png")
print("- aircraft_crosswake_animation.gif")