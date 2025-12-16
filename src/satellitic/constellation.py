#!/usr/bin/env python3
"""
constellation.py

Generate scalable Walker-style satellite constellations and export TLEs
from pandas DataFrame definitions.

Dependencies:
    pip install numpy pandas sgp4
"""
from .init import *

import numpy as np
import pandas as pd
from datetime import datetime
from sgp4.api import Satrec

MU = MU_EARTH_GRAV
R_EARTH = R_EARTH_KM

# ---------------------------------------------------------------------
# Orbital utilities
# ---------------------------------------------------------------------

def mean_motion_rev_per_day(alt_km: float) -> float:
    """
    Convert circular-orbit altitude to mean motion (rev/day).
    """
    a = R_EARTH + alt_km
    n_rad_s = np.sqrt(MU / a**3)
    return n_rad_s * 86400.0 / (2.0 * np.pi)

# ---------------------------------------------------------------------
# TLE generation
# ---------------------------------------------------------------------

def generate_constellation_tles(
    df: pd.DataFrame,
    satnum_start: int = 10000,
    eccentricity: float = 1e-4,
    argp_deg: float = 0.0,
    bstar: float = 0.0
) -> pd.DataFrame:
    """
    Generate TLEs for multiple constellation systems defined in a DataFrame.

    Required DataFrame columns:
        system
        height_km
        n_planes
        sats_per_plane
        inclination_deg
        raan0_deg

    Returns:
        DataFrame with one row per satellite:
            system, satnum, plane, slot, tle1, tle2
    """

    tles = []
    satnum = satnum_start
    epoch = datetime.utcnow()

    # Epoch in TLE fractional day-of-year format
    epoch_day = (
        (epoch - datetime(epoch.year, 1, 1)).days + 1
        + (epoch.hour + epoch.minute / 60 + epoch.second / 3600) / 24
    )

    for _, row in df.iterrows():
        n_planes = int(row.n_planes)
        sats_per_plane = int(row.sats_per_plane)

        mean_motion = mean_motion_rev_per_day(row.height_km)
        mean_motion_rad_min = mean_motion * 2.0 * np.pi / 1440.0

        for p in range(n_planes):
            raan_deg = (row.raan0_deg + p * 360.0 / n_planes) % 360.0

            for s in range(sats_per_plane):
                mean_anomaly_deg = (s * 360.0 / sats_per_plane) % 360.0

                sat = Satrec()
                sat.sgp4init(
                    0,              # WGS84
                    'i',            # improved SGP4
                    satnum,
                    epoch_day,
                    bstar,
                    0.0, 0.0,       # ndot, nddot
                    eccentricity,
                    np.radians(argp_deg),
                    np.radians(row.inclination_deg),
                    np.radians(mean_anomaly_deg),
                    mean_motion_rad_min,
                    np.radians(raan_deg)
                )

                tle1, tle2 = sat.export_tle()

                tles.append({
                    "system": row.system,
                    "satnum": satnum,
                    "plane": p,
                    "slot": s,
                    "tle1": tle1,
                    "tle2": tle2
                })

                satnum += 1

    return pd.DataFrame(tles)

# ---------------------------------------------------------------------
# Example / Test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example constellation definitions (Systems A–C)
    constellation_df = pd.DataFrame([
        # system, height_km, n_planes, sats_per_plane, inclination_deg, raan0_deg
        ["A", 525, 28, 120, 53.0, 0.0],
        ["B", 610, 36, 36, 42.0, 0.0],
        ["RT", 1200, 36, 40, 88.0, 0.0],
    ], columns=[
        "system",
        "height_km",
        "n_planes",
        "sats_per_plane",
        "inclination_deg",
        "raan0_deg"
    ])

    print("Generating TLEs...")
    tle_df = generate_constellation_tles(
        constellation_df,
        satnum_start=20000
    )

    print(f"Generated {len(tle_df)} satellites\n")

    # Show first few TLEs
    for _, row in tle_df.head(3).iterrows():
        print(row.tle1)
        print(row.tle2)
        print()

    # Optional: write to file
    output_file = "constellation.tle"
    with open(output_file, "w") as f:
        for _, row in tle_df.iterrows():
            f.write(row.tle1 + "\n")
            f.write(row.tle2 + "\n")

    print(f"TLEs written to {output_file}")
