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
import numpy as np

R_EARTH = 6371e3  # meter

# ------------------------------
# Poisson-verktyg för osäkerhet
# ------------------------------
def poisson_cdf(k, lam) :
    """
    CDF för Poisson(lam) vid heltal k (inklusive k).
    Summation med rekursion i termen; OK för små-medelstora lam.
    """
    if k < 0:
        return 0.0
    # För stora lam, normalapprox (kontrollerat fall-back)
    if lam > 1e5:
        # kontinuerlig normalapprox med continuity correction
        mu = lam
        sigma = math.sqrt(lam)
        z = (k + 0.5 - mu) / sigma
        # standard normal CDF
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # Exakt summation
    term = math.exp(-lam)  # P(0)
    cdf = term
    for n in range(1, k + 1):
        term *= lam / n
        cdf += term
        # Tidig avbryt om vi redan nått ~1.0 (numerisk stabilitet)
        if 1.0 - cdf < 1e-15:
            return 1.0
    return cdf

def poisson_ppf(p, lam):
    """
    Invertera CDF: minsta heltal k så att CDF(k) >= p.
    Binärsökning med en rimlig övre gräns; normalapprox för mycket stora lam.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p måste ligga i [0,1].")
    if p == 0.0:
        return 0
    if p == 1.0:
        # praktiskt taget "oändlig" – välj en konservativ övre gräns
        return int(max(10, lam + 10 * math.sqrt(max(lam, 1.0))) + 50)

    # Normalapprox som snabb gissning för stora lam
    if lam > 1e5:
        mu = lam
        sigma = math.sqrt(lam)
        # normal PPF ~ mu + z_p*sigma, begränsa till >=0
        # z för p erhålls via approximativ inverserf – här grovt via bisektion runt -6..6
        # men för enkelhet: använd en liten binärsökning i heltal direkt på CDF.
        pass  # vi faller tillbaka till binärsökning även här

    # Hitta övre gräns dynamiskt
    k_hi = int(max(10, lam + 10 * math.sqrt(max(lam, 1.0))) + 10)
    while poisson_cdf(k_hi, lam) < p:
        k_hi *= 2
        if k_hi > 10**7:  # säkerhetsbroms
            break
    k_lo = 0
    # Binärsökning
    while k_lo < k_hi:
        k_mid = (k_lo + k_hi) // 2
        c = poisson_cdf(k_mid, lam)
        if c >= p:
            k_hi = k_mid
        else:
            k_lo = k_mid + 1
    return k_lo

def poisson_ci_95(lam) :
    """
    95%-intervall på counts för Poisson(lam): [k_2.5%, k_97.5%].
    (Exakt via PPF/CDF-inversion för små-medelstora lam; normalapprox i CDF för mycket stora lam.)
    """
    lower = poisson_ppf(0.025, lam)
    upper = poisson_ppf(0.975, lam)
    return lower, upper


def antenna_gain(theta):
    """
    Enkel antennmodell
    """
    G0 = 10**(35/10)      # 35 dBi
    theta_3db = np.deg2rad(2.5)
    return G0 * np.exp(-(theta/theta_3db)**2)

def random_unit_vectors(n):
    """Uniforma riktningar på en sfär"""
    u = np.random.rand(n)
    v = np.random.rand(n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    return np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))

def angle_between(a, b):
    """Vinkel mellan två vektorer"""
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))

def fspl_weight(r):
    """Avståndsviktning för EPFD"""
    return (R_EARTH / r)**2

def exclusion_mask(directions, axis, theta_ex):
    """
    True = tillåten att sända
    """
    angles = angle_between(directions, axis)
    return angles > theta_ex

def beam_mask(directions, earth_axis, theta_beam):
    """
    True = jordstationen ligger i strålningskonen
    """
    angles = angle_between(directions, earth_axis)
    return angles < theta_beam

def cosd(angle_deg) :
    return math.cos(math.radians(angle_deg))

def sind(angle_deg) :
    return math.sin(math.radians(angle_deg))

def p_beam(theta_beam_deg: float) -> float:
    """
    Geometrisk sannolikhet att en stråle träffar stationen:
    p_beam = (1 - cos(theta_beam))/2  (vinkel i grader)
    """
    return (1.0 - cosd(theta_beam_deg)) / 2.0

def theta_ex_of_elevation(eps):
    """
    Konservativ modell:
    större exklusion vid låg elevation
    """
    return np.deg2rad(5 + 15*np.exp(-eps/np.deg2rad(15)))

def _resolve_theta_ex_for_pair(
    j, i, eps_deg, theta_ex_default = theta_ex_of_elevation ,
    theta_ex_map = None ):
    """
    Hämta theta_ex^{j->i}(eps) i grader.
    - Om (j,i) finns i theta_ex_map används det (kan vara konstant eller funktion av eps).
    - Annars används theta_ex_default (även den kan vara konstant eller funktion).
    """
    if theta_ex_map and (j, i) in theta_ex_map:
        th = theta_ex_map[(j, i)]
    else:
        th = theta_ex_default
    return th(eps_deg) if callable(th) else float(th)

def p_on_for_shell_index_elev(
    i, shell_order, eps_deg, theta_ex_default = theta_ex_of_elevation ,
    theta_ex_map = None ):
    """
    p_on^{(i)}(eps) = Prod_{j>i} (1 + cos(theta_ex^{j->i}(eps)))/2
    där j>i innebär "högre" skal (större orbitalradie).
    """
    p_on = 1.0
    pos_i = shell_order.index(i)
    for pos_j in range(pos_i + 1, len(shell_order)):
        j = shell_order[pos_j]
        theta_ex_ji = _resolve_theta_ex_for_pair(j, i, eps_deg, theta_ex_default, theta_ex_map)
        p_on *= (1.0 + cosd(theta_ex_ji)) / 2.0
    return p_on

def psi_max_from_elevation(e, R_E, r):
    """
    Maximum geocentric angle psi (rad) for a given elevation e (rad)
    """
    Re_r = R_E / r
    term = np.sqrt(1 - (Re_r*np.cos(e))**2)
    cos_psi = Re_r*np.cos(e)**2 + np.sin(e)*term
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    return np.arccos(cos_psi)

def draw_epfd(mask_level=-150):
    import matplotlib.pyplot as plt

    epfd_mean,epfd99,elevations = epfd_means_shells(     shells = [
        dict(N=1600, h=550e3, P=5),    # LEO
        dict(N=400,  h=1200e3, P=10),  # högre LEO
    ] )

    plt.figure()
    plt.plot(np.rad2deg(elevations), 10*np.log10(epfd_mean),
         label="Mean EPFD")

    # EPFD-mask
    epfd_mask = mask_level * np.ones_like(epfd_mean)
    plt.plot(np.rad2deg(elevations), epfd_mask,
         "--", label="EPFD-mask")

    plt.xlabel("Elevation angle (degrees)")
    plt.ylabel("EPFD (dBW/m²/Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def epfd_means_shells(     shells = [
        dict(N=1600, h=550e3 , P=5),    # LEO
        dict(N=400,  h=1200e3, P=10),   # högre LEO
    ], elevations = np.deg2rad(np.linspace(0, 90, 90)) ):
    print("TODO : Check means at some point.")

    epfd_mean = []
    epfds = []
    for eps in elevations:
        epfd_eps = 0.0
        for s in shells:
            r = R_EARTH + s["h"]
            shell_epfd = epfd_shell(
                N=s["N"],
                r=r,
                P=s["P"],
                G_func=antenna_gain,
                theta_ex=theta_ex_of_elevation(eps),
                theta_beam=np.deg2rad(8)
            )
            #print(eps,shell_epfd)
            epfd_eps += np.abs(shell_epfd)
        epfds.append(shell_epfd)
        epfd_mean.append(epfd_eps)
    epfd_99 = np.percentile(epfds, 99)
    epfd_mean = np.array(epfd_mean)
    return epfd_mean, epfd_99, elevations


def epfd_shell (
    N, r, P, G_func,
    theta_ex ,                      # degrees
    elevation_min_deg = 8 ,           # degrees
    theta_beam = 8 ,                # degrees
    R_E = 6371e3,
    elevation_angle = None ,        # optional: radians
    n_quad=64 ,
    epfd_floor = 1e-20,             # linear units floor to avoid log(0)
    smooth_transition_deg = 2       # smooth low-angle transition in degrees
    ):
    theta_beam      = np.deg2rad(theta_beam)
    theta_ex_deg    = theta_ex
    theta_ex        = np.deg2rad(theta_ex)
    
    """
    Compute the mean EPFD from a satellite shell with smooth low-elevation behaviour.

    Parameters
    ----------
    N : int
        Number of satellites in the shell.
    r : float
        Orbital radius (m).
    P : float
        Transmit power spectral density (W/Hz).
    G_func : callable
        Antenna gain function of off-axis angle (radians).
    theta_ex : float
        Exclusion cone angle (radians).
    theta_beam : float
        Beamwidth angle toward Earth (radians).
    elevation_angle : float or None
        Optional elevation angle for smooth low-angle weighting (radians).
    n_quad : int
        Number of Gauss–Legendre points.
    epfd_floor : float
        Linear EPFD floor to avoid negative/zero in dB.
    smooth_transition_deg : float
        Smoothing width for low-elevation transition (degrees).

    Returns
    -------
    EPFD : float
        Linear EPFD (W/m^2/Hz)
    """
    from numpy.polynomial.legendre import leggauss
    # 1. Gauss–Legendre quadrature over theta
    x, w = leggauss(n_quad)
    theta = 0.5*(theta_beam - theta_ex)*x + 0.5*(theta_beam + theta_ex)
    weights = 0.5*(theta_beam - theta_ex)*w

    # 2. Integrand (antenna gain × sin(theta))
    integrand = G_func(theta) * np.sin(theta)
    integral = np.sum(integrand * weights)

    Nco = Nvis_Nco_shell_with_exclusion( N=N, r=r,
                    elevation_min_deg = elevation_min_deg ,
                    theta_ex_deg = theta_ex_deg ,
                    R_E = 6371e3 )['Nco']

    # 3. Compute linear EPFD
    epfd_lin = Nco * P * integral / (4*np.pi*r**2)

    # 4. Optional: smooth low-elevation weighting (numerical smoothing)
    if elevation_angle is not None:
        # Smooth transition function W(e)
        e_deg = np.rad2deg(elevation_angle)
        e_c = smooth_transition_deg  # center of smooth turn-on
        delta_e = smooth_transition_deg / 2
        W = 1.0 / (1 + np.exp(-(e_deg - e_c)/delta_e))
        epfd_lin *= W

    # 5. Ensure non-negative EPFD
    epfd_lin = np.clip(epfd_lin, epfd_floor, None)

    return epfd_lin

# ------------------------------
# Huvudberäkningar för N_co med elevation
# ------------------------------
def compute_Nco_for_elevation(
    shells, eps_deg, theta_beam_deg,
    theta_ex_default = theta_ex_of_elevation,
    p_freq = 1.0, theta_ex_map = None ) :
    """
    Beräkna N_co per skal och totalt för given elevation eps_deg.

    shells: lista av dicts:
      {
        "id": valfri etikett (int/str),
        "r": orbitalradie [m],  # r = R_EARTH + h
        "N": antal satelliter i skalet
      }

    theta_beam_deg: halvvinkel (deg) för effektiv strålning som kan belysa stationen.
    theta_ex_default: antingen konstant (grad) eller funktion f(eps_deg)->grad.
    p_freq: spektralt överlapp (B_overlap / B_tot), i [0,1].
    theta_ex_map: valfri mapping {(j,i): ThetaSpec} där ThetaSpec=konstant eller f(eps)->grad.
                  Här är j ett "högre" skal än i (större r).
    """
    if not 0.0 <= p_freq <= 1.0:
        raise ValueError("p_freq måste ligga i intervallet [0, 1].")
    if theta_beam_deg < 0:
        raise ValueError("theta_beam_deg måste vara icke-negativ.")
    # Sortera skal enligt växande r (lägre -> högre) för att kunna tolka j>i
    idx_sorted = sorted(range(len(shells)), key=lambda k: shells[k]["r"])
    p_b = p_beam(theta_beam_deg)

    per_shell = []
    N_co_total_mean = 0.0
    for i_idx in range(len(shells)):
        i = i_idx
        p_on = p_on_for_shell_index_elev(i, idx_sorted, eps_deg, theta_ex_default, theta_ex_map)
        N_i = shells[i]["N"]
        lam_i = N_i * p_on * p_b * p_freq  # λ_i = E[N_co^{(i)}]
        per_shell.append({
            "id": shells[i].get("id", i),
            "r": shells[i]["r"],
            "N": N_i,
            "p_on": p_on,
            "p_beam": p_b,
            "p_freq": p_freq,
            "lambda": lam_i,                      # medelvärde och varians
            "var": lam_i,
            "std": math.sqrt(lam_i),
            "ci95": poisson_ci_95(lam_i)
        })
        N_co_total_mean += lam_i

    # Summa av oberoende Poisson ≈ Poisson(sum λ_i) (antagande i texten)
    total = {
        "lambda": N_co_total_mean,
        "var": N_co_total_mean,
        "std": math.sqrt(N_co_total_mean),
        "ci95": poisson_ci_95(N_co_total_mean)
    }

    return {"eps_deg": eps_deg, "per_shell": per_shell, "total": total}

def compute_pfreq(f_victim, f_interferer):
    """
    f_victim = (f_min_v, f_max_v)
    f_interferer = (f_min_i, f_max_i)
    Frekvenser i Hz eller GHz, så länge de matchar.
    """
    f_vmin, f_vmax = f_victim
    f_imin, f_imax = f_interferer

    overlap = max(0.0, min(f_vmax, f_imax) - max(f_vmin, f_imin))
    B_tot = f_vmax - f_vmin  # eller interfererbandet, beroende på modell

    return overlap / B_tot if B_tot > 0 else 0.0

def compute_Nco_elevation_sweep (
        shells, eps_grid_deg, theta_beam_deg,
        theta_ex_default = theta_ex_of_elevation,
        p_freq = 1.0, theta_ex_map = None ) :
    """
    Kör beräkning för flera elevationsvinklar och returnera en lista med resultat per elevation.
    """
    results = []
    for eps in eps_grid_deg:
        results.append(
            compute_Nco_for_elevation(
                shells, eps, theta_beam_deg, theta_ex_default, p_freq, theta_ex_map
            )
        )
    return results

# LEGACY
def Nvis_Nco_shell_with_exclusion(
    N, r,
    elevation_min_deg ,
    theta_ex_deg ,
    R_E = 6371e3
):
    """
    Mean visible and co-channel satellites for one NGSO shell,
    including GSO exclusion cone statistics.
    """
    # --- Visibility (e >= 0)
    f_vis = (1 - R_E/r) * 0.5
    Nvis  = N * f_vis

    # --- Elevation-limited
    e_min = np.deg2rad(elevation_min_deg)
    psi_e = psi_max_from_elevation(e_min, R_E, r)
    f_elev = (1 - np.cos( psi_e )) / 2

    # --- GSO exclusion. Cone is from satellite to earth
    theta_ex = np.deg2rad( theta_ex_deg )
    f_allowed = (1 + np.cos(theta_ex)) / 2

    # --- Co-channel
    Nco = Nvis * f_elev * f_allowed

    return {'Nvis':Nvis, 'Nco':Nco}

def anpassning_total_Nco ( Nsat ) :
   return 0.004859*Nsat

if __name__=='__main__' :
    draw_epfd()
