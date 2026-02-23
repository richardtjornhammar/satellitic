lic_ = """
   Copyright 2026 Richard Tjörnhammar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as xp

def slant_range(Re, h, e, xp):
    """
    Re : Earth radius
    h  : satellite altitude
    e  : elevation angle [rad]
    """
    rs = Re + h
    return -Re * xp.sin(e) + xp.sqrt(rs**2 - (Re * xp.cos(e))**2)


def central_angle(Re, h, e, xp):
    """
    Earth central angle ψ corresponding to elevation e
    """
    rs = Re + h
    return xp.arccos((Re / rs) * xp.cos(e))

def subsatellite_point(r_sat, xp):
    """
    r_sat : (3,) satellite position in ECEF
    returns (lat, lon) in radians
    """
    x, y, z = r_sat

    lon = xp.arctan2(y, x)
    r_xy = xp.sqrt(x**2 + y**2)
    lat = xp.arctan2(z, r_xy)

    return lat, lon


def visibility_mask(lat_grid, lon_grid,
                    lat_s, lon_s,
                    psi_max,
                    xp):
    """
    lat_grid, lon_grid : 2D arrays (radians)
    lat_s, lon_s       : satellite subpoint
    psi_max            : max central angle
    """

    sin_lat = xp.sin(lat_grid)
    cos_lat = xp.cos(lat_grid)

    sin_phi_s = xp.sin(lat_s)
    cos_phi_s = xp.cos(lat_s)

    cospsi = (
        sin_phi_s * sin_lat +
        cos_phi_s * cos_lat * xp.cos(lon_grid - lon_s)
    )

    return cospsi >= xp.cos(psi_max)


def elevation_angle(r_sat, r_ground, xp):
    """
    r_sat    : (3,) satellite ECEF
    r_ground : (...,3) ground points ECEF

    Returns elevation angle in radians
    """

    rho = r_sat - r_ground
    rho_norm = xp.linalg.norm(rho, axis=-1)

    rho_hat = rho / rho_norm[..., None]

    # Ground normal vector
    n_hat = r_ground / xp.linalg.norm(r_ground, axis=-1)[..., None]

    sin_e = xp.sum(rho_hat * n_hat, axis=-1)

    return xp.arcsin(sin_e)


def latlon_to_ecef(lat, lon, Re, xp):
    """
    lat, lon can be arrays
    Returns (...,3)
    """
    cos_lat = xp.cos(lat)
    return xp.stack([
        Re * cos_lat * xp.cos(lon),
        Re * cos_lat * xp.sin(lon),
        Re * xp.sin(lat)
    ], axis=-1)


def regular_grid_clipping( lat , lon , lat_s , lon_s , psi_max ) :
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    # Bounding box in angular space
    lat_min = lat_s - psi_max
    lat_max = lat_s + psi_max

    lon_min = lon_s - psi_max
    lon_max = lon_s + psi_max

    i_min = int((lat_min - lat[0]) / dlat)
    i_max = int((lat_max - lat[0]) / dlat)

    j_min = int((lon_min - lon[0]) / dlon)
    j_max = int((lon_max - lon[0]) / dlon)

    return(i_min,i_max,j_min,j_max)

def geodetic_to_ecef(lat, lon, h, model, xp):
    a  = model["a"]
    e2 = model["e2"]

    sin_lat = xp.sin(lat)
    cos_lat = xp.cos(lat)

    N = a / xp.sqrt(1 - e2 * sin_lat**2)

    x = (N + h) * cos_lat * xp.cos(lon)
    y = (N + h) * cos_lat * xp.sin(lon)
    z = (N * (1 - e2) + h) * sin_lat

    return xp.stack([x, y, z], axis=-1)

def ecef_to_geodetic(r, model, xp, max_iter=5):
    a  = model["a"]
    e2 = model["e2"]

    x, y, z = r[...,0], r[...,1], r[...,2]

    lon = xp.arctan2(y, x)
    p   = xp.sqrt(x**2 + y**2)

    lat = xp.arctan2(z, p * (1 - e2))

    for _ in range(max_iter):
        sin_lat = xp.sin(lat)
        N = a / xp.sqrt(1 - e2 * sin_lat**2)
        h = p / xp.cos(lat) - N
        lat = xp.arctan2(z, p * (1 - e2 * N/(N+h)))

    return lat, lon, h


WGS84 = {
    "a": 6378137.0,                # semi-major axis [m]
    "f": 1.0 / 298.257223563       # flattening
}
WGS84["b"]  = WGS84["a"] * (1 - WGS84["f"])
WGS84["e2"] = 1 - (WGS84["b"]**2 / WGS84["a"]**2)

def elevation_angle_ellipsoid(r_sat, r_ground, xp):
    rho = r_sat - r_ground
    rho_norm = xp.linalg.norm(rho, axis=-1)

    rho_hat = rho / rho_norm[...,None]

    # Surface normal from ellipsoid gradient
    x, y, z = r_ground[...,0], r_ground[...,1], r_ground[...,2]
    a = WGS84["a"]
    b = WGS84["b"]

    n = xp.stack([
        x/(a*a),
        y/(a*a),
        z/(b*b)
    ], axis=-1)

    n_hat = n / xp.linalg.norm(n, axis=-1)[...,None]

    sin_e = xp.sum(rho_hat * n_hat, axis=-1)

    return xp.arcsin(sin_e)

def subsatellite_latlon(
        r_sat,
        earth_center=None,
        model=None,
        xp=None):
    """
    r_sat : (3,) satellite position in ECEF [m]
    earth_center : (3,) optional Earth center offset
    model : WGS84 dictionary
    xp : numpy or jax.numpy

    Returns:
        lat (rad), lon (rad)
    """

    if xp is None:
        import numpy as xp

    if model is None:
        model = WGS84

    if earth_center is None:
        earth_center = xp.zeros(3)

    # Shift into Earth-centered frame
    r = r_sat - earth_center

    a = model["a"]
    b = model["b"]

    x, y, z = r

    # ---- Ray from origin through satellite ----
    # Solve intersection with ellipsoid:
    #
    # (x t)^2 / a^2 + (y t)^2 / a^2 + (z t)^2 / b^2 = 1
    #
    # Solve for t

    denom = (
        (x*x + y*y) / (a*a) +
        (z*z)       / (b*b)
    )

    t = 1.0 / xp.sqrt(denom)

    # Intersection point
    x_i = x * t
    y_i = y * t
    z_i = z * t

    # ---- Convert to geodetic ----
    lon = xp.arctan2(y_i, x_i)

    p = xp.sqrt(x_i**2 + y_i**2)
    e2 = model["e2"]

    # Iterative latitude solve (fast convergence)
    lat = xp.arctan2(z_i, p*(1 - e2))

    for _ in range(5):
        sin_lat = xp.sin(lat)
        N = a / xp.sqrt(1 - e2*sin_lat**2)
        lat = xp.arctan2(z_i + e2*N*sin_lat, p)

    return lat, lon



def slant_range_vec(r_sat, r_ground, xp):
    return xp.linalg.norm(r_sat - r_ground, axis=-1)


def compute_pfd_grid(
        r_sat,
        lat_grid,
        lon_grid,
        P_tx,
        gain_function,
        xp):

    # Convert grid to ECEF
    r_ground = geodetic_to_ecef(
        lat_grid, lon_grid, 0.0,
        WGS84, xp
    )

    # Slant range
    rho = slant_range_vec(r_sat, r_ground, xp)

    # Off-nadir angle
    sat_nadir = -r_sat / xp.linalg.norm(r_sat)
    rho_vec = r_ground - r_sat
    rho_hat = rho_vec / rho[...,None]

    cos_theta = xp.sum(sat_nadir * rho_hat, axis=-1)
    theta = xp.arccos(xp.clip(cos_theta, -1, 1))

    G = gain_function(theta)

    PFD = (P_tx * G) / (4 * xp.pi * rho**2)

    return PFD

# Example Gain Model (simple circular beam)
def circular_beam(theta, theta_3db, xp):
    return xp.where(
        theta <= theta_3db,
        1.0,
        0.0
    )

def integrate_surface(PFD, lat_grid, dlat, dlon, xp):
    a  = WGS84["a"]
    e2 = WGS84["e2"]

    sin_lat = xp.sin(lat_grid)
    denom = xp.sqrt(1 - e2 * sin_lat**2)

    dA = (a*a * xp.cos(lat_grid) / denom) * dlat * dlon

    return xp.sum(PFD * dA)

#ARTICLE 21 REQUIRES SUM ALL VISIBLE SPACE STATIONS
PFD_total = xp.sum(PFD_satellites, axis=0)

# PFD_max_over_grid[t] = xp.max(PFD_dB)
# percentile_01 = xp.percentile(PFD_max_over_grid, 99.9)
# limit = pfd(delta) from article 21
# percentile_01 <= limit # compliant at 0.1% level.

"""
ITU defines PFD as:

PFD = Pt*G(θ)/( 4πρ**2*B )
Pt = total transmit power [W]
G(θ) = antenna gain toward Earth point
ρ = slant range
B = reference bandwidth (Hz)

for time in simulation:
    for each satellite:
        compute PFD grid
    sum satellites
    compute worst-case PFD
    store max value
evaluate percentiles
compare to Article 21 table
"""


if __name__ == '__main__' :
    Re = 6371e3
    h  = 550e3
    e_min = xp.deg2rad(10.0)

    # Satellite ECEF position
    r_sat = xp.array([Re + h, 0.0, 0.0])

    # Subpoint
    lat_s, lon_s = subsatellite_point(r_sat, xp)

    # Central angle
    psi_max = central_angle(Re, h, e_min, xp)

    # Build grid
    lat = xp.linspace(-xp.pi/2, xp.pi/2, 361)
    lon = xp.linspace(-xp.pi, xp.pi, 721)
    lat_grid, lon_grid = xp.meshgrid(lat, lon, indexing="ij")

    # Compute mask
    mask = visibility_mask(lat_grid, lon_grid,
                       lat_s, lon_s,
                       psi_max, xp)

