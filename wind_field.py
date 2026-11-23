# Simple 3D Simulator of Low Level Turbulence (LLT)

# import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Computational domain
x_min, x_max = -50.0, 300.0
y_min, y_max = -120.0, 120.0
z_min, z_max = 0.0, 220.0

nx, ny, nz = 120, 80, 60

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Animation runs from 0 to 60 s
dt = 0.5
animation_duration = 60.0
n_steps = int(animation_duration / dt) + 1
time = np.arange(n_steps) * dt

# Static output times
static_times = [10.0, 30.0, 60.0, 300.0]

# Time used for the static shear section
shear_time = 10.0


# Input background Wind Field
# Assuming the wind is constant in space and time 
U0 = 10.0  # m/s
# Later use different U0 values for different scenarios

# Width, Height, and Length of the building 
building_width = 30.0  # m , y direction
building_height = 100.0 # m, z direction
building_length = 30.0 # m, x direction

building_mask = (
    (X >= 0.0)
    & (X <= building_length)
    & (np.abs(Y) <= building_width / 2.0)
    & (Z <= building_height)
)

# Idealised building-wake

def generate_flow(X, Y, Z, t, U0=10.0):

    # Distance downstream from the rear face of the building
    x_downstream = np.maximum(X - building_length, 0.0)

    # Only activate wake downstream of the building
    downstream_mask = X > building_length

    # Wake width grows downstream
    sigma_y = 0.5 * building_width + 0.08 * x_downstream

    # Wake vertical thickness grows more slowly
    sigma_z = 0.35 * building_height + 0.015 * x_downstream

    # Wake centre height
    z_centre = 0.55 * building_height

    # Gaussian wake envelope
    wake_envelope = (
        np.exp(-x_downstream / 180.0)
        * np.exp(-Y**2 / (2.0 * sigma_y**2))
        * np.exp(-(Z - z_centre)**2 / (2.0 * sigma_z**2))
    )

    # Remove the wake upstream of the building
    wake_envelope *= downstream_mask


    # Streamwise velocity deficit
 
    # Time-dependent idealised wake
    deficit_strength = 0.65

    # Approximate vortex-shedding frequency
    Strouhal_number = 0.2
    shedding_frequency = Strouhal_number * U0 / building_width

    phase = 2.0 * np.pi * shedding_frequency * t

    # Approximate downstream convection speed and wavelength
    convection_speed = 0.7 * U0
    wake_wavelength = convection_speed / shedding_frequency

    # Travelling wake phase:
    # constant phase moves downstream as time increases
    travelling_phase = (
        phase
        - 2.0 * np.pi * x_downstream / wake_wavelength
    )

    # Weak low-frequency modulation:
    # makes t = 30, 60 and 300 s visually different
    slow_modulation = 1.0 + 0.08 * np.sin(
        2.0 * np.pi * t / 173.0
    )

    # Wake deficit changes weakly with time
    wake_modulation = (
        slow_modulation
        * (1.0 + 0.12 * np.sin(travelling_phase))
    )

    # Streamwise velocity
    u = U0 * (
        1.0
        - deficit_strength * wake_envelope * wake_modulation
    )

    # Oscillatory cross-wind disturbance
    v_amplitude = 0.35 * U0

    v = (
        v_amplitude
        * wake_envelope
        * slow_modulation
        * np.sin(travelling_phase)
    )

    # Idealised vertical upwash and lee-side downwash
    w_amplitude = 0.15 * U0

    # Upward motion near the windward roof region
    updraft = np.exp(
        -((X - 0.0)**2) / (2.0 * 18.0**2)
        -((Z - building_height)**2) / (2.0 * 20.0**2)
    )

    # Downward motion in the lee-side wake
    downwash = np.exp(
        -((X - (building_length + 35.0))**2) / (2.0 * 30.0**2)
        -((Z - 0.65 * building_height)**2) / (2.0 * 28.0**2)
    )

    # Keep vertical motion concentrated near the building centreline
    vertical_lateral_envelope = np.exp(
        -(Y**2) / (2.0 * (0.7 * building_width)**2)
    )

    vertical_modulation = (
        slow_modulation
        * (1.0 + 0.15 * np.sin(travelling_phase))
    )

    w = (
        w_amplitude
        * (updraft - 0.8 * downwash)
        * vertical_lateral_envelope
        * vertical_modulation
    )

    return u, v, w


def calculate_diagnostics(u, v, w):

    speed = np.sqrt(u**2 + v**2 + w**2)

    # Grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # Derivatives
    du_dx, du_dy, du_dz = np.gradient(u, dx, dy, dz)
    dv_dx, dv_dy, dv_dz = np.gradient(v, dx, dy, dz)
    dw_dx, dw_dy, dw_dz = np.gradient(w, dx, dy, dz)

    vertical_shear = np.sqrt(
        du_dz**2 +
        dv_dz**2 +
        dw_dz**2
    )
    vertical_shear[building_mask] = np.nan

    # General velocity-gradient magnitude
    gradient_magnitude = np.sqrt(
        du_dx**2 + du_dy**2 + du_dz**2
        + dv_dx**2 + dv_dy**2 + dv_dz**2
        + dw_dx**2 + dw_dy**2 + dw_dz**2
    )
    vertical_vorticity = dv_dx - du_dy

    return {
        "speed": speed,
        "vertical_shear": vertical_shear,
        "gradient_magnitude": gradient_magnitude,
        "vertical_vorticity": vertical_vorticity,
    }


if __name__ == "__main__":

    z_view = 30.0
    z_index = np.argmin(np.abs(z - z_view))

    y_view = 0.0
    y_index = np.argmin(np.abs(y - y_view))

    arrow_step_x_xy = 5
    arrow_step_y_xy = 5

    arrow_step_x_xz = 6
    arrow_step_z_xz = 3

    building_x = [0, building_length, building_length, 0, 0]
    building_y = [
        -building_width / 2,
        -building_width / 2,
        building_width / 2,
        building_width / 2,
        -building_width / 2
    ]

    # x-z centre-plane flow plot

    def draw_xz_flow(ax, t):

        ax.clear()

        u, v, w = generate_flow(X, Y, Z, t=t, U0=U0)

        # Extract y = 0 vertical plane
        u_xz = u[:, y_index, :]
        w_xz = w[:, y_index, :]

        # Only show components within the x-z plane
        speed_xz = np.sqrt(u_xz**2 + w_xz**2)

        building_xz = building_mask[:, y_index, :]

        speed_xz = np.where(building_xz, np.nan, speed_xz)
        u_xz = np.where(building_xz, np.nan, u_xz)
        w_xz = np.where(building_xz, np.nan, w_xz)

        mesh = ax.pcolormesh(
            X[:, y_index, :],
            Z[:, y_index, :],
            speed_xz,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1.3 * U0
        )

        ax.quiver(
            X[::arrow_step_x_xz, y_index, ::arrow_step_z_xz],
            Z[::arrow_step_x_xz, y_index, ::arrow_step_z_xz],
            u_xz[::arrow_step_x_xz, ::arrow_step_z_xz],
            w_xz[::arrow_step_x_xz, ::arrow_step_z_xz],
            color="white",
            scale=180,
            width=0.002,
            angles="uv"
        )

        # Building silhouette in the vertical plane
        ax.fill(
            [0, building_length, building_length, 0],
            [0, 0, building_height, building_height],
            color="dimgray",
            alpha=0.95,
            label="Idealised building"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)

        ax.set_xlabel("x downstream (m)")
        ax.set_ylabel("Height z (m)")

        ax.set_title(
            f"Vertical centre-plane flow at y = {y_view:.1f} m, "
            f"t = {t:.1f} s"
        )

        ax.legend(loc="upper right")

        return mesh

    # x-y horizontal flow plot
    
    def draw_xy_flow(ax, t):

        ax.clear()

        u, v, w = generate_flow(X, Y, Z, t=t, U0=U0)

        # Full 3D speed, displayed in the selected horizontal layer
        speed = np.sqrt(u**2 + v**2 + w**2)

        speed_xy = speed[:, :, z_index]
        u_xy = u[:, :, z_index]
        v_xy = v[:, :, z_index]

        building_xy = building_mask[:, :, z_index]

        speed_xy = np.where(building_xy, np.nan, speed_xy)
        u_xy = np.where(building_xy, np.nan, u_xy)
        v_xy = np.where(building_xy, np.nan, v_xy)

        mesh = ax.pcolormesh(
            X[:, :, z_index],
            Y[:, :, z_index],
            speed_xy,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=1.3 * U0
        )

        ax.quiver(
            X[::arrow_step_x_xy, ::arrow_step_y_xy, z_index],
            Y[::arrow_step_x_xy, ::arrow_step_y_xy, z_index],
            u_xy[::arrow_step_x_xy, ::arrow_step_y_xy],
            v_xy[::arrow_step_x_xy, ::arrow_step_y_xy],
            color="white",
            scale=180,
            width=0.002
        )

        ax.plot(
            building_x,
            building_y,
            color="red",
            linewidth=3,
            label="Building footprint"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("x downstream (m)")
        ax.set_ylabel("y cross-wind (m)")

        ax.set_title(
            f"Horizontal wind field at z = {z_view:.1f} m, "
            f"t = {t:.1f} s"
        )

        ax.legend(loc="upper right")

        return mesh

    # x-z animation

    fig_xz_animation, ax_xz_animation = plt.subplots(figsize=(12, 6))

    initial_xz_mesh = draw_xz_flow(
        ax_xz_animation,
        time[0]
    )

    fig_xz_animation.colorbar(
        initial_xz_mesh,
        ax=ax_xz_animation,
        label=r"In-plane wind speed $\sqrt{u^2+w^2}$ (m s$^{-1}$)"
    )

    def update_xz(frame):
        draw_xz_flow(ax_xz_animation, time[frame])

    xz_animation = FuncAnimation(
        fig_xz_animation,
        update_xz,
        frames=len(time),
        interval=80,
        blit=False
    )

    fig_xz_animation.tight_layout()

    xz_animation.save(
        "xz_flow_animation.gif",
        writer="pillow",
        fps=12
    )

    plt.close(fig_xz_animation)

    # Static x-z snapshots
    
    for t_static in static_times:

        fig, ax = plt.subplots(figsize=(12, 6))

        mesh = draw_xz_flow(ax, t_static)

        fig.colorbar(
            mesh,
            ax=ax,
            label=r"In-plane wind speed $\sqrt{u^2+w^2}$ (m s$^{-1}$)"
        )

        fig.tight_layout()

        fig.savefig(
            f"xz_flow_t{int(t_static)}s.png",
            dpi=300
        )

        plt.close(fig)

    # x-y animation

    fig_xy_animation, ax_xy_animation = plt.subplots(figsize=(12, 6))

    initial_xy_mesh = draw_xy_flow(
        ax_xy_animation,
        time[0]
    )

    fig_xy_animation.colorbar(
        initial_xy_mesh,
        ax=ax_xy_animation,
        label=r"Wind speed $|\mathbf{V}|$ (m s$^{-1}$)"
    )

    def update_xy(frame):
        draw_xy_flow(ax_xy_animation, time[frame])

    xy_animation = FuncAnimation(
        fig_xy_animation,
        update_xy,
        frames=len(time),
        interval=80,
        blit=False
    )

    fig_xy_animation.tight_layout()

    xy_animation.save(
        "xy_flow_animation.gif",
        writer="pillow",
        fps=12
    )

    plt.close(fig_xy_animation)

    # Static x-y snapshots
    
    for t_static in static_times:

        fig, ax = plt.subplots(figsize=(12, 6))

        mesh = draw_xy_flow(ax, t_static)

        fig.colorbar(
            mesh,
            ax=ax,
            label=r"Wind speed $|\mathbf{V}|$ (m s$^{-1}$)"
        )

        fig.tight_layout()

        fig.savefig(
            f"xy_flow_t{int(t_static)}s.png",
            dpi=300
        )

        plt.close(fig)

    print("All x-z and x-y animations and static figures were saved.")