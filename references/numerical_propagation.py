"""
Testing out NumericalPropagator, comparing against KeplerianPropagator
"""

import orekit_jpype
orekit_jpype.initVM()
from orekit_jpype.pyhelpers import setup_orekit_data
setup_orekit_data()

# OREKIT Java imports
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory

# Python imports
import matplotlib.pyplot as plt
import numpy as np

# ============================================================= #
# ORBIT DEFINITION
# ============================================================= #

# epoch
utc = TimeScalesFactory.getUTC()
epoch = AbsoluteDate(2026, 1, 23, 11, 39, 0.0, utc)

# orbital params (keplerian)
a = 7000e3
e = 0.001
i = np.radians(10)
raan = np.radians(45)
omega = np.radians(0)
nu = np.radians(0)

true_anomaly_type = getattr(PositionAngleType, "TRUE")

orbit = KeplerianOrbit(
    a,                                      # semi major axis (m)
    e,                                      # eccentricity
    i,                                      # inclination (radians)
    omega,                                  # arg perigee (radians)
    raan,                                   # RAAN (radians)
    nu,                                     # true anomaly (radians)
    getattr(PositionAngleType, "TRUE"),     # define true anomaly (mean anomaly also possible)
    FramesFactory.getEME2000(),             # inertial frame
    epoch,
    3.986004418e14                          # Earth's gravitational param (m^3/s^2)
)

# orbit information
period_seconds = 2 * np.pi / orbit.getKeplerianMeanMotion()
periapsis = orbit.getA() * (1-orbit.getE())
apoapsis = orbit.getA() * (1+orbit.getE())

print(f"\nOrbital Period:         {period_seconds / 60:.2f} minutes")
print(f"Perigee Altitude:       {(periapsis - 6371000) / 1000:.2f} km")
print(f"Apogee Altitude:        {(apoapsis - 6371000) / 1000:.2f} km")

# ============================================================= #
# KEPLERIAN PROPAGATION
# ============================================================= #
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import IERSConventions, Constants

# init analytical propagator
kep_prop = KeplerianPropagator(orbit)

# define propagation parameters
num_orbits_to_propagate = 2
prop_duration = period_seconds * num_orbits_to_propagate
timestep = 60

# init storage
time_list_analytical = []
lat_list_analytical = []
lon_list_analytical = []
x_list_analytical = []
y_list_analytical = []
z_list_analytical = []

# create Earth Ellipsoid for LatLon extraction later
earth_ellipsoid = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    FramesFactory.getITRF(IERSConventions.IERS_2010, True) # ECEF without tidal waves
)

# propagate
current_time = 0.0 # seconds
while current_time <= prop_duration:
    # get orbital state at current time
    elapsed_time = epoch.shiftedBy(current_time)
    propagated_orbit = kep_prop.propagate(elapsed_time)

    # get ICRF position
    position_icrf = propagated_orbit.getPVCoordinates().getPosition()
    x_list_analytical.append(position_icrf.getX() / 1000.0)  # Convert to km
    y_list_analytical.append(position_icrf.getY() / 1000.0)
    z_list_analytical.append(position_icrf.getZ() / 1000.0)

    # transform to ITRF frame (ECEF)
    itrf_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True) # ECEF without tidal waves
    position_itrf = propagated_orbit.getPVCoordinates(itrf_frame).getPosition()
    # get latlon
    geodetic_point = earth_ellipsoid.transform(position_itrf, itrf_frame, elapsed_time)
    lat_list_analytical.append(np.degrees(geodetic_point.getLatitude()))
    lon_list_analytical.append(np.degrees(geodetic_point.getLongitude()))

    # collect and advance time
    time_list_analytical.append(current_time/ 3600.0) # hours
    current_time += timestep

# ============================================================= #
# NUMERICAL PROPAGATION
# ============================================================= #

# Numerical propagator imports
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.gravity import Relativity
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import CelestialBodyFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# create initial spacecraft state
initial_state = SpacecraftState(orbit)
initial_state = initial_state.withMass(1000.0)  # 1000 kg satellite

# configure ODE integrator (Dormand-Prince 8(5,3))
min_step = 0.001   # 1 ms minimum step
max_step = 60.0    # 60 s maximum step
pos_tolerance = 0.01  # 1 m position tolerance
int_tolerance = 1e-6
integrator = DormandPrince853Integrator(min_step, max_step, pos_tolerance, int_tolerance)

# create numerical propagator with just integrator
num_prop = NumericalPropagator(integrator)
num_prop.resetInitialState(initial_state)

# add force models
# 1. Gravity (degree/order 70x70 for high-fidelity)
gravity_field = GravityFieldFactory.getNormalizedProvider(70, 70)
# gravity_field = GravityFieldFactory.getNormalizedProvider(1, 0)  # Use only for comparison (central force)
gravity_force = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravity_field)
num_prop.addForceModel(gravity_force)
print(f"  Gravity: ({gravity_field.getMaxDegree()}, {gravity_field.getMaxOrder()})")

# 2. Third-body perturbations (Sun)
sun = CelestialBodyFactory.getSun()
third_body_sun = ThirdBodyAttraction(sun)
num_prop.addForceModel(third_body_sun)
print("  Third-body: Sun")

# 3. Third-body perturbations (Moon)
moon = CelestialBodyFactory.getMoon()
third_body_moon = ThirdBodyAttraction(moon)
num_prop.addForceModel(third_body_moon)
print("  Third-body: Moon")

# 4. Atmospheric Drag (Harris-Priester Model)
# Harris-Priester is a simple, density-dependent model suitable for LEO (100-1000 km)
# Uses embedded density table from Montenbruck & Gill for mean solar activity
atmosphere = HarrisPriester(sun, earth_ellipsoid)

# Define satellite drag properties
# IsotropicDrag(draggedArea, dragCoefficient)
# For a 1000 kg satellite with ~10 m^2 reference area and Cd ~2.2
drag_force = DragForce(atmosphere, IsotropicDrag(10.0, 2.2))
num_prop.addForceModel(drag_force)
print("  Atmospheric Drag: Harris-Priester (10 m², Cd=2.2)")

# 5. Solar Radiation Pressure (SRP)
# Note: SRP requires RadiationSensitive provider and ExtendedPositionProvider setup
# For now, commented out as it requires additional configuration
# TODO: Implement with proper RadiationSensitive object
# srp = SolarRadiationPressure(...)
# num_prop.addForceModel(srp)

# 6. Relativity (General Relativity corrections)
# Important for precise orbit determination, ~mm level effect
try:
    from org.orekit.forces.gravity import Relativity as RelativityCorrection
    relativity = RelativityCorrection(Constants.WGS84_EARTH_MU)
    num_prop.addForceModel(relativity)
    print("  Relativity corrections: enabled")
except:
    print("  Relativity corrections: not available in this OREKIT version")

# init storage for numerical propagation
time_list_numerical = []
lat_list_numerical = []
lon_list_numerical = []
x_list_numerical = []
y_list_numerical = []
z_list_numerical = []

# propagate
print("\nPropagating with Numerical Propagator...")
current_time = 0.0
while current_time <= prop_duration:
    # get orbital state at current time
    elapsed_time = epoch.shiftedBy(current_time)
    propagated_state = num_prop.propagate(elapsed_time)
    propagated_orbit = propagated_state.getOrbit()

    # get ICRF position
    position_icrf = propagated_orbit.getPVCoordinates().getPosition()
    x_list_numerical.append(position_icrf.getX() / 1000.0)  # Convert to km
    y_list_numerical.append(position_icrf.getY() / 1000.0)
    z_list_numerical.append(position_icrf.getZ() / 1000.0)

    # transform to ITRF frame (ECEF)
    position_itrf = propagated_orbit.getPVCoordinates(itrf_frame).getPosition()
    # get latlon
    geodetic_point = earth_ellipsoid.transform(position_itrf, itrf_frame, elapsed_time)
    lat_list_numerical.append(np.degrees(geodetic_point.getLatitude()))
    lon_list_numerical.append(np.degrees(geodetic_point.getLongitude()))

    # collect and advance time
    time_list_numerical.append(current_time / 3600.0)  # hours
    current_time += timestep

print("Numerical propagation complete.")
print(f"  Total time steps: {len(time_list_numerical)}")


# ============================================================= #
# PROPAGATOR COMPARISON
# ============================================================= #

pos_num = np.array([
    x_list_numerical,
    y_list_numerical,
    z_list_numerical
])

pos_kep = np.array([
    x_list_analytical,
    y_list_analytical,
    z_list_analytical
])

dist = np.linalg.norm(pos_num - pos_kep, axis=0)

plt.figure()
plt.plot(lon_list_analytical, lat_list_analytical, 'b', label='analytic', alpha=0.5)
plt.plot(lon_list_numerical, lat_list_numerical, 'r', label='numerical', alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Satellite Grountracks per propagator')
plt.grid(True)
plt.legend()

plt.figure()
plt.grid(True)
plt.plot(time_list_analytical, dist)
plt.xlabel('Simulation time [hrs]')
plt.ylabel('Distance [km]')
plt.title('Satellite Position Divergence over Time')