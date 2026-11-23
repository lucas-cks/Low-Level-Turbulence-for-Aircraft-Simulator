import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from wind_field import (
    x, y, z,
    X, Y, Z,
    U0,
    building_length,
    building_height,
    generate_flow
)

# Virtual Doppler-LIDAR configuration
# Place the LIDAR just downstream of the building.
# It scans further downstream into the lee-side wake.
x_lidar = 35.0
y_lidar = 0.0
z_lidar = 5.0

# RHI: fixed azimuth / fixed y-plane, varying elevation angle.
ranges = np.linspace(5.0, 240.0, 180)       # m
elevation_deg = np.linspace(2.0, 50.0, 90)  # degrees
elevation = np.deg2rad(elevation_deg)

# Shape: (number of elevation angles, number of range gates)
R, THETA = np.meshgrid(ranges, elevation)

# Beam coordinates in the x-z vertical plane.
X_beam = x_lidar + R * np.cos(THETA)
Y_beam = np.full_like(X_beam, y_lidar)
Z_beam = z_lidar + R * np.sin(THETA)

beam_points = np.column_stack(
    (
        X_beam.ravel(),
        Y_beam.ravel(),
        Z_beam.ravel()
    )
)

# Times corresponding to the static wind-field plots
scan_times = [10.0, 30.0, 60.0, 300.0]

for scan_time in scan_times:

    # Generate the true simulated atmospheric field

    u, v, w = generate_flow(
        X,
        Y,
        Z,
        t=scan_time,
        U0=U0
    )

    # Interpolate u and w onto virtual RHI beam locations

    u_interpolator = RegularGridInterpolator(
        (x, y, z),
        u,
        bounds_error=False,
        fill_value=np.nan
    )

    w_interpolator = RegularGridInterpolator(
        (x, y, z),
        w,
        bounds_error=False,
        fill_value=np.nan
    )

    u_beam = u_interpolator(
        beam_points
    ).reshape(X_beam.shape)

    w_beam = w_interpolator(
        beam_points
    ).reshape(X_beam.shape)

    # Calculate synthetic Doppler radial velocity

    # Beam unit vector:
    # e_r = (cos(theta), 0, sin(theta))
    #
    # Doppler radial velocity:
    # Vr = u cos(theta) + w sin(theta)

    Vr = (
        u_beam * np.cos(THETA)
        + w_beam * np.sin(THETA)
    )

    # Uniform-wind background signal
    Vr_background = U0 * np.cos(THETA)

    # Difference from uniform background flow
    Vr_anomaly = Vr - Vr_background

    # Use a symmetric colour range for positive/negative anomaly
    anomaly_limit = np.nanmax(np.abs(Vr_anomaly))

    # Avoid a zero colour range if an ideal test case has no anomaly
    if anomaly_limit == 0:
        anomaly_limit = 0.1

    # Plot the virtual RHI scan

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        sharey=True
    )

    # Left panel: raw radial velocity
    
    raw_plot = axes[0].pcolormesh(
        X_beam,
        Z_beam,
        Vr,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=U0
    )

    axes[0].scatter(
        x_lidar,
        z_lidar,
        color="red",
        s=70,
        marker="^",
        zorder=5,
        label="Virtual LIDAR"
    )

    axes[0].fill(
        [0, building_length, building_length, 0],
        [0, 0, building_height, building_height],
        color="dimgray",
        alpha=0.9,
        label="Idealised building"
    )

    axes[0].set_xlabel("x downstream (m)")
    axes[0].set_ylabel("Height z (m)")
    axes[0].set_title("Synthetic RHI Doppler radial velocity")

    axes[0].set_xlim(-10, 300)
    axes[0].set_ylim(0, 220)
    axes[0].legend(loc="upper right")

    fig.colorbar(
        raw_plot,
        ax=axes[0],
        label=r"Radial velocity $V_r$ (m s$^{-1}$)"
    )

    # Right panel: anomaly from uniform background
    
    anomaly_plot = axes[1].pcolormesh(
        X_beam,
        Z_beam,
        Vr_anomaly,
        shading="auto",
        cmap="RdBu_r",
        vmin=-anomaly_limit,
        vmax=anomaly_limit
    )

    axes[1].scatter(
        x_lidar,
        z_lidar,
        color="black",
        s=70,
        marker="^",
        zorder=5,
        label="Virtual LIDAR"
    )

    axes[1].fill(
        [0, building_length, building_length, 0],
        [0, 0, building_height, building_height],
        color="dimgray",
        alpha=0.9,
        label="Idealised building"
    )

    axes[1].set_xlabel("x downstream (m)")
    axes[1].set_title(
        "Radial-velocity anomaly relative to uniform flow"
    )

    axes[1].set_xlim(-10, 300)
    axes[1].set_ylim(0, 220)
    axes[1].legend(loc="upper right")

    fig.colorbar(
        anomaly_plot,
        ax=axes[1],
        label=r"$V_r-U_0\cos(\theta)$ (m s$^{-1}$)"
    )

    fig.suptitle(
        f"Virtual RHI scan at t = {scan_time:.0f} s",
        fontsize=16
    )

    fig.tight_layout()

    output_name = f"synthetic_rhi_t{int(scan_time)}s.png"

    fig.savefig(
        output_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_name}")