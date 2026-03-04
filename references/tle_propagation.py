"""
OREKIT-JPype example: Propagate satellite from TLE (Two-Line Element)
Shows how to read TLE data and propagate orbits using SGP4 model
"""

# Initialize OREKIT first before any Java imports
import orekit_jpype
orekit_jpype.initVM()

from orekit_jpype.pyhelpers import setup_orekit_data
import math

# Setup OREKIT data
setup_orekit_data()

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.frames import FramesFactory
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions, Constants

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# TLE DATA
# ============================================================================
# Get TLEs from:
# - https://celestrak.org/ (free, no registration)
# - https://www.space-track.org/ (free with registration)

# Example: ISS (International Space Station)
print("=" * 70)
print("TLE-based Satellite Propagation using OREKIT-JPype")
print("=" * 70)

tle_line1 = "1 25544U 98067A   26022.05647685  .00011470  00000-0  22186-3 0  9997"
tle_line2 = "2 25544  51.6397 160.2505 0001651  64.7750 295.3734 15.54288662434352"

# Create TLE object
tle = TLE(tle_line1, tle_line2)

print(f"\nSatellite Information:")
print(f"  Satellite Number: {tle.getSatelliteNumber()}")
print(f"  Inclination: {tle.getI() * 180 / 3.14159265359:.2f}°")
print(f"  RAAN: {tle.getRaan() * 180 / 3.14159265359:.2f}°")
print(f"  Argument of Perigee: {tle.getPerigeeArgument() * 180 / 3.14159265359:.2f}°")
print(f"  Mean Motion: {tle.getMeanMotion():.6f} rad/s")
print(f"  Eccentricity: {tle.getE():.6f}")

# Create SGP4 propagator from TLE
propagator = TLEPropagator.selectExtrapolator(tle)

# Get the orbit and epoch from TLE propagator
epoch = tle.getDate()
orbit = propagator.getInitialState().getOrbit()

print(f"\nOrbital Elements (from TLE):")
print(f"  Epoch: {epoch}")
print(f"  Semi-major axis: {orbit.getA() / 1000:.2f} km")
print(f"  Eccentricity: {orbit.getE():.6f}")
print(f"  Inclination: {orbit.getI() * 180 / 3.14159265359:.2f}°")

# ============================================================================
# PROPAGATION
# ============================================================================

print("\n" + "=" * 70)
print("Propagating satellite for 2 orbital periods...")
print("=" * 70)

# Calculate orbital period
orbital_period_seconds = 2 * 3.14159265359 / orbit.getKeplerianMeanMotion()
prop_duration = orbital_period_seconds * 2

# Sample interval
sample_interval = 60.0

# Storage for results
time_list = []
lat_list = []
lon_list = []
x_list = []
y_list = []
z_list = []

# Earth ellipsoid for geodetic conversion
earth_ellipsoid = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    FramesFactory.getITRF(IERSConventions.IERS_2010, True)
)

# Propagate
current_time = 0.0
while current_time <= prop_duration:
    elapsed_time = epoch.shiftedBy(current_time)
    propagated_orbit = propagator.propagate(elapsed_time)

    # Get ICRF position
    position_icrf = propagated_orbit.getPVCoordinates().getPosition()
    x_list.append(position_icrf.getX() / 1000.0)
    y_list.append(position_icrf.getY() / 1000.0)
    z_list.append(position_icrf.getZ() / 1000.0)

    # Get geodetic coordinates
    itrf_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    pv_itrf = propagated_orbit.getPVCoordinates(itrf_frame)
    position_itrf = pv_itrf.getPosition()

    geodetic_point = earth_ellipsoid.transform(position_itrf, itrf_frame, elapsed_time)

    lat_rad = geodetic_point.getLatitude()
    lon_rad = geodetic_point.getLongitude()

    time_list.append(current_time / 3600.0)
    lat_list.append(math.degrees(lat_rad))
    lon_list.append(math.degrees(lon_rad))

    current_time += sample_interval

print(f"Propagation complete. Collected {len(time_list)} data points")

# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(18, 7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Ground track
ax1.plot(lon_list, lat_list, 'b-', linewidth=2, label='Ground Track')
ax1.scatter(lon_list[0], lat_list[0], color='green', s=100, marker='o', label='Start', zorder=5)
ax1.scatter(lon_list[-1], lat_list[-1], color='red', s=100, marker='x', label='End', zorder=5)
ax1.set_xlabel('Longitude (degrees)', fontsize=12)
ax1.set_ylabel('Latitude (degrees)', fontsize=12)
ax1.set_title(f'TLE-based Ground Track - {prop_duration/3600:.2f} hours', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)

# 3D orbit
ax2.plot(x_list, y_list, z_list, 'r-', linewidth=2, label='Satellite Orbit')
ax2.scatter(x_list[0], y_list[0], z_list[0], color='green', s=100, marker='o', label='Start', zorder=5)
ax2.scatter(x_list[-1], y_list[-1], z_list[-1], color='red', s=100, marker='x', label='End', zorder=5)
ax2.set_xlabel('X (km)', fontsize=12)
ax2.set_ylabel('Y (km)', fontsize=12)
ax2.set_zlabel('Z (km)', fontsize=12)
ax2.set_title(f'TLE-based Orbit - ICRF Frame (3D)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Earth sphere
earth_radius = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
earth_x = (earth_radius/1000) * np.outer(np.cos(u), np.sin(v))
earth_y = (earth_radius/1000) * np.outer(np.sin(u), np.sin(v))
earth_z = (earth_radius/1000) * np.outer(np.ones(np.size(u)), np.cos(v))
ax2.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='blue', edgecolor='none')

max_range = 7000
ax2.set_xlim(-max_range, max_range)
ax2.set_ylim(-max_range, max_range)
ax2.set_zlim(-max_range, max_range)

plt.tight_layout()
plt.savefig('tle_propagation_plots.png', dpi=150, bbox_inches='tight')
print(f"\nPlots saved to: tle_propagation_plots.png")
plt.show()

print("=" * 70)
print("Propagation Statistics:")
print(f"  Total duration: {prop_duration/3600:.2f} hours")
print(f"  Number of data points: {len(time_list)}")
print(f"  Latitude range: {min(lat_list):.2f}° to {max(lat_list):.2f}°")
print(f"  Longitude range: {min(lon_list):.2f}° to {max(lon_list):.2f}°")
print("=" * 70)
