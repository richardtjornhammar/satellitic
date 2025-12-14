lic_ = """
   Copyright 2025 Richard Tjörnhammar

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

import math

# ---------------------------------------------------------
# 1. Ange AP4-värden här 
# ---------------------------------------------------------

apog_km = 500.0        # höjd över jordytan (km)
perig_km = 300.0       # höjd över jordytan (km)
incl_deg = 55.0        # inclin_ang
raan_deg = 120.0       # right_asc eller long_asc
argp_deg = 30.0        # perig_arg
true_anom_deg = 0.0    # sann anomali vid den tidpunkt du vill räkna position

# ---------------------------------------------------------
# 2. Konstanter
# ---------------------------------------------------------

R_earth = 6378.137               # km
mu = 398600.4418                 # km^3/s^2

# ---------------------------------------------------------
# 3. Härledda Kepler-element
# ---------------------------------------------------------

# Räknar om höjd över jordytan till avstånd från centrum
r_apo = R_earth + apog_km
r_per = R_earth + perig_km

# Semimajor axis
a = (r_apo + r_per) / 2.0

# Excentricitet
e = (r_apo - r_per) / (r_apo + r_per)

# Omloppstid (Keplers 3:e lag)
T = 2 * math.pi * math.sqrt(a**3 / mu)

# Medelrörelse
n = 2 * math.pi / T   # rad/s

# ---------------------------------------------------------
# 4. Position & hastighet i perifokal-ram (PQW)
# ---------------------------------------------------------

# Konvertera till radianer
i = math.radians(incl_deg)
Omega = math.radians(raan_deg)
omega = math.radians(argp_deg)
nu = math.radians(true_anom_deg)

# Avstånd vid sann anomali
r = a * (1 - e**2) / (1 + e * math.cos(nu))

# Position i PQW
x_p = r * math.cos(nu)
y_p = r * math.sin(nu)
z_p = 0.0

# Hastighet i PQW
vx_p = -math.sqrt(mu / (a * (1 - e**2))) * math.sin(nu)
vy_p =  math.sqrt(mu / (a * (1 - e**2))) * (e + math.cos(nu))
vz_p = 0.0

# ---------------------------------------------------------
# 5. Omvandla PQW → ECI (klassisk 3-rotation)
# ---------------------------------------------------------

def rotate_PQW_to_ECI(xp, yp, zp):
    # Rotation: Rz(-Omega) * Rx(-i) * Rz(-omega)
    sinO = math.sin(Omega); cosO = math.cos(Omega)
    sini = math.sin(i);     cosi = math.cos(i)
    sinw = math.sin(omega); cosw = math.cos(omega)

    R11 =  cosO*cosw - sinO*sinw*cosi
    R12 = -cosO*sinw - sinO*cosw*cosi
    R13 =  sinO*sini
    R21 =  sinO*cosw + cosO*sinw*cosi
    R22 = -sinO*sinw + cosO*cosw*cosi
    R23 = -cosO*sini
    R31 =  sinw*sini
    R32 =  cosw*sini
    R33 =  cosi

    X = R11*xp + R12*yp + R13*zp
    Y = R21*xp + R22*yp + R23*zp
    Z = R31*xp + R32*yp + R33*zp

    return X, Y, Z

r_eci = rotate_PQW_to_ECI(x_p, y_p, z_p)
v_eci = rotate_PQW_to_ECI(vx_p, vy_p, vz_p)

# ---------------------------------------------------------
# 6. Utskrift av resultat
# ---------------------------------------------------------

print("\n--- ORBITAL ELEMENT RESULTAT ---")
print(f"Semimajor axis a:         {a:.3f} km")
print(f"Eccentricitet e:          {e:.6f}")
print(f"Omloppstid T:             {T/60:.3f} minuter")
print(f"Mean motion:              {n:.8f} rad/s  ({86400/T:.6f} rev/dag)")

print("\n--- STATE VECTOR (ECI) ---")
print(f"Position (km): X={r_eci[0]:.3f}, Y={r_eci[1]:.3f}, Z={r_eci[2]:.3f}")
print(f"Hastighet (km/s): VX={v_eci[0]:.6f}, VY={v_eci[1]:.6f}, VZ={v_eci[2]:.6f}")

