"""
Physics-based wrinkle prediction for steered prepreg tape (thermoset AFP).

Criterion (per path point, per fibre across the tape width):

    wrinkle  <=>  compressive strain  -eps(s, y)  >=  eps_crit(s)
    eps_crit = max(eps_bif, eps_Gamma)      evaluated just behind the roller
                                            ("immediate") and after tack
                                            relaxation ("delayed")

    eps_bif   = 2*sqrt(D_eff*K) / (E1*h)        buckling on the tack foundation
    eps_Gamma = sqrt(2*Gamma / (E1*h))          energy to peel the tape off

Wrinkle geometry (post-buckling estimate):
    length    L   = 2*pi*(D_eff/K)^(1/4)                     buckling wavelength
    spacing   gap = -l*ln(1 - eps_crit/eps),  l = sqrt(E1*h/k_s)
              (after a wrinkle, the compression rebuilds through the tack by shear lag)
    height    H   = (2/pi)*sqrt(L*dL),  dL = tape length released into one wrinkle

Model chain:
    geometry  -> fibre strain eps(s, y) from the 3D tape surface (centreline + normals)
    roller    -> contact length, pressure, dwell time      (Winkler roller layer)
    heater    -> nip temperature, cooling history          (hot-gas convection)
    tack      -> intimate contact D_ic (Lee-Springer), healing D_h,
                 tack energy Gamma (Gent-Schultz), foundation stiffness K(t)
                 (Prony series + Arrhenius time-temperature shift),
                 optional spatial variability (lognormal random field)

Default material/tack values are PLACEHOLDERS (CYCOM 977-2 order of magnitude,
partly from Kheradpisheh PhD 2025). Calibrate them with probe-tack, relaxation
and viscosity tests before quantitative use.

Usage:
    python wrinkle_predictor.py
All settings are hard-coded: edit the run settings and PARAMS below.
RUN selects the case: "s_course", "arc_1500" or "sensitivity" (heating x speed study).

Path CSV columns: x, y, z [PATH_UNITS], optional nx, ny, nz (surface normal),
optional per-point process: speed_mm_s, force_N, T_nip_C, T_gas_C, tension_N.
"""

import copy
import csv
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ----------------------------------------------------------------------------- run settings

PATH_CSV = None            # path CSV file (used instead of the built-in case when set)
PATH_UNITS = "mm"          # units of x, y, z in PATH_CSV: "mm" or "m"
# built-in 'S' course: straight, constant-radius left turn, right turn with growing radius
DEMO_FLAT_M = 0.3          # straight length
DEMO_ARC1_RADIUS_M = 0.1   # first turn radius (constant)
DEMO_ARC1_ANGLE_DEG = 90.0 # first turn angle
DEMO_ARC2_R_START_M = 0.1  # second (opposite) turn: radius at its start ...
DEMO_ARC2_R_END_M = 2.0    # ... growing linearly to this radius
DEMO_ARC2_LENGTH_M = 0.55  # second turn length
# built-in constant-curvature case: 90 deg sector of R 1500 mm, 6.35 x 0.185 mm tape,
# with spatial variability of the tack (spaced wrinkle areas near the threshold)
ARC_RADIUS_M = 2.0
ARC_ANGLE_DEG = 90.0
ARC_WIDTH_M = 6.35e-3
ARC_THICKNESS_M = 0.185e-3
ARC_TACK_COV = 0.3         # coefficient of variation of the local tack (K, Gamma, k_s)
ARC_TACK_CORR_LENGTH_M = 0.05
ARC_TACK_SEED = 3
# one-point calibration of the tack level (K, Gamma, k_s scaled together) so that the median
# minimum steerable radius equals this value at the base process; None keeps the default values
ARC_TARGET_R_CRIT_M = 1.3
# sensitivity study (on the arc case): heating level x deposition speed
SENS_T_GAS_C = list(range(60, 301, 20))        # hot gas temperature, low -> high
SENS_SPEEDS_MM_S = [25, 50, 76, 125, 200]      # head speed (sets the compaction dwell time)
SENS_CSV = "sensitivity_results.csv"
SENS_PNG = "sensitivity.png"

OUT_CSV = "wrinkle_results.csv"
WIDTH_SCALE = 10.0         # tape width exaggeration in the 3D view
HEIGHT_SCALE = 20.0        # wrinkle height exaggeration in the 3D view
SCREENSHOT = None          # PNG file to save the 3D view instead of opening a window
SHOW_PLOT = True

R_GAS = 8.314  # J/(mol K)
C2K = 273.15

PARAMS = {
    "material": {
        "E1_Pa": 31e9,             # effective axial modulus of the uncured tape
        "E_bend_Pa": 31e9,         # modulus giving the uncured bending stiffness D_eff = E_bend*h^3/12
        "thickness_m": 0.17e-3,
        "width_m": 6.35e-3,
        "rho_kg_m3": 1580.0,
        "cp_J_kgK": 1000.0,
    },
    "resin": {
        "eta_ref_Pas": 1.0e4,      # viscosity at T_ref
        "T_ref_C": 45.0,
        "Ea_J_mol": 70e3,          # flow activation energy, also used as the time-temperature shift
    },
    "tack": {
        "D0": 0.3,                 # initial degree of intimate contact
        "C_ic": 20.0,              # Lee-Springer lumped roughness constant
        "t_heal_ref_s": 0.05,      # healing time at T_ref
        "Gamma0_J_m2": 6.0,        # tack energy at full contact, zero rate
        "phi_ref": 2.0,            # viscoelastic dissipation factor at T_ref and v_ref
        "phi_exp": 0.25,           # Gent-Schultz rate exponent
        "v_ref_m_s": 0.076,
        "k_ref_Pa_m": 1.0e10,      # foundation stiffness at full contact, t=0, T_ref
        "prony_inf_frac": 0.34,    # K_inf / K_0   (thesis Kz(t) ratio)
        "prony_terms": [[0.66, 43.5]],   # [fraction, tau_s] at T_ref
        "k_shear_Pa_m": 1.0e9,     # in-plane interface shear stiffness (shear lag -> wrinkle spacing)
        "level": 1.0,              # global tack multiplier on K, Gamma and k_s (calibration)
    },
    "variability": {               # lognormal spatial field (mean 1) multiplying K, Gamma and k_s
        "cov": 0.0,
        "correlation_length_m": 0.05,
        "seed": 1,
    },
    "thermal": {
        "heater": "hot_gas",       # "hot_gas" or "nip_temperature"
        "T_gas_C": 160.0,
        "h_conv_W_m2K": 120.0,
        "heated_length_m": 0.03,
        "T_nip_C": 45.0,           # used when heater == "nip_temperature"
        "T_initial_C": 23.0,
        "T_tool_C": 23.0,
        "h_tool_W_m2K": 500.0,     # conduction into the tool after the nip
    },
    "roller": {
        "radius_m": 0.019,
        "contact_width_m": 0.0254,  # width loaded by the roller (e.g. 4 tows)
        "layer_stiffness_Pa_m": 4e9,  # ~4e9 polyurethane 60A, ~3e11 steel roller
    },
    "process": {
        "speed_m_s": 0.076,
        "force_N": 250.0,
        "tension_N": 0.0,
    },
    "model": {
        "resample_step_m": 1e-3,
        "smoothing_window_m": 0.01,
        "n_width": 21,
        "strain_reference": "equilibrium",  # "equilibrium" (mean strain = tension) or "fibre"
        "neutral_axis_offset_m": 0.0,        # used with "fibre": unstrained fibre offset from centreline
        "evaluation_time_s": 600.0,          # delayed check (e.g. time until the next ply covers it)
        "delayed_opening_rate_m_s": 1e-4,
    },
}


# ----------------------------------------------------------------------------- geometry

def demo_path(flat_length, arc1_radius, arc1_angle_deg, arc2_radius_start, arc2_radius_end,
              arc2_length, step=5e-3):
    """Built-in 'S' course on a flat tool [m].

    1. straight segment of flat_length
    2. left turn at constant radius arc1_radius over arc1_angle_deg
    3. right turn whose radius grows linearly from arc2_radius_start to arc2_radius_end
       over arc2_length
    """
    arc1_length = arc1_radius * np.radians(arc1_angle_deg)
    length = flat_length + arc1_length + arc2_length
    s = np.linspace(0.0, length, int(round(length / step)) + 1)

    kappa = np.zeros_like(s)
    in_arc1 = (s > flat_length) & (s <= flat_length + arc1_length)
    kappa[in_arc1] = 1.0 / arc1_radius
    in_arc2 = s > flat_length + arc1_length
    frac = (s[in_arc2] - flat_length - arc1_length) / arc2_length
    kappa[in_arc2] = -1.0 / (arc2_radius_start + frac * (arc2_radius_end - arc2_radius_start))

    ds = np.diff(s, prepend=0.0)
    heading = np.cumsum(kappa * ds)
    x = np.cumsum(np.cos(heading) * ds)
    y = np.cumsum(np.sin(heading) * ds)
    pts = np.column_stack([x, y, np.zeros_like(s)])
    normals = np.tile([0.0, 0.0, 1.0], (len(s), 1))
    return pts, normals, {}


def arc_path(radius, angle_deg, step=5e-3):
    """Built-in constant-curvature course on a flat tool: angular sector of the given radius [m]."""
    length = radius * np.radians(angle_deg)
    phi = np.linspace(0.0, np.radians(angle_deg), int(round(length / step)) + 1)
    pts = np.column_stack([radius * np.sin(phi), radius * (1.0 - np.cos(phi)), np.zeros_like(phi)])
    return pts, np.tile([0.0, 0.0, 1.0], (len(phi), 1)), {}


def read_path_csv(path, units):
    scale = {"m": 1.0, "mm": 1e-3}[units]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty path file: {path}")
    cols = {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}
    pts = np.column_stack([cols["x"], cols["y"], cols["z"]]) * scale
    normals = (np.column_stack([cols["nx"], cols["ny"], cols["nz"]])
               if {"nx", "ny", "nz"} <= cols.keys() else None)
    proc_cols = {k: cols[k] for k in ("speed_mm_s", "force_N", "T_nip_C", "T_gas_C", "tension_N")
                 if k in cols}
    return pts, normals, proc_cols


def resample(pts, normals, proc_cols, step):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    pts = pts[keep]
    normals = normals[keep] if normals is not None else None
    proc_cols = {k: v[keep] for k, v in proc_cols.items()}

    s_in = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = max(int(np.ceil(s_in[-1] / step)) + 1, 5)
    s = np.linspace(0.0, s_in[-1], n)
    interp = lambda a: np.column_stack([np.interp(s, s_in, a[:, k]) for k in range(a.shape[1])])
    p = interp(pts)
    if normals is None:
        nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    else:
        nrm = interp(normals)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    proc = {k: np.interp(s, s_in, v) for k, v in proc_cols.items()}
    return s, p, nrm, proc


def fibre_strains(s, p, nrm, params):
    """Axial strain of each fibre across the width, from the 3D tape surface.

    Fibre j sits at r_j(s) = c(s) + y_j d(s), d = n x t.  Its length rate |c' + y_j d'|
    captures in-plane (geodesic) curvature to first order and twist to second order.
    """
    mat, mdl = params["material"], params["model"]
    b = mat["width_m"]
    sigma = max(mdl["smoothing_window_m"] / (s[1] - s[0]) / 2.0, 1e-6)
    smooth = lambda a: gaussian_filter1d(a, sigma, axis=0, mode="nearest")

    dc = smooth(np.gradient(p, s, axis=0))
    t = dc / np.linalg.norm(dc, axis=1, keepdims=True)
    nrm = smooth(nrm)
    nrm -= np.sum(nrm * t, axis=1, keepdims=True) * t
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    d = np.cross(nrm, t)
    dd = smooth(np.gradient(d, s, axis=0))

    y = np.linspace(-b / 2, b / 2, mdl["n_width"])
    rate = np.linalg.norm(dc[:, None, :] + y[None, :, None] * dd[:, None, :], axis=2)  # (N, M)

    if mdl["strain_reference"] == "fibre":
        ref = np.linalg.norm(dc + mdl["neutral_axis_offset_m"] * dd, axis=1)
    else:  # force equilibrium: mean strain across the width equals the tension strain
        ref = rate.mean(axis=1)
    eps = rate / ref[:, None] - 1.0

    ddc = smooth(np.gradient(t, s, axis=0))
    geo = {
        "kappa_g": np.sum(ddc * d, axis=1),   # in-plane curvature [1/m], >0 turning towards +d
        "twist": np.sum(dd * nrm, axis=1),    # twist rate [rad/m]
        "tangent": t, "normal": nrm, "side": d,
    }
    return y, eps, geo


# ----------------------------------------------------------------------------- process physics

def arrhenius_shift(T_C, resin):
    """a_T = eta(T)/eta(T_ref); used for viscosity, healing and relaxation times."""
    T = np.asarray(T_C) + C2K
    return np.exp(resin["Ea_J_mol"] / R_GAS * (1.0 / T - 1.0 / (resin["T_ref_C"] + C2K)))


def tack_field(s, params):
    """Spatial variability of the local tack quality (resin distribution, contact, substrate):
    lognormal random field of mean 1, given CoV and correlation length."""
    var = params["variability"]
    if var["cov"] <= 0.0:
        return np.ones_like(s)
    rng = np.random.default_rng(var["seed"])
    g = gaussian_filter1d(rng.standard_normal(len(s)), var["correlation_length_m"] / (s[1] - s[0]),
                          mode="wrap")
    g /= g.std()
    sig = np.sqrt(np.log(1.0 + var["cov"] ** 2))
    return np.exp(sig * g - 0.5 * sig ** 2)


def interface_state(speed, force, tension, T_nip, params, tack_factor=1.0):
    """Roller contact, thermal history and tack properties for arrays of points."""
    mat, resin, tack = params["material"], params["resin"], params["tack"]
    th, rol, mdl = params["thermal"], params["roller"], params["model"]
    E1h = mat["E1_Pa"] * mat["thickness_m"]
    D_eff = mat["E_bend_Pa"] * mat["thickness_m"] ** 3 / 12.0

    # roller as a Winkler layer: F/w = k * (4/3) a^3 / (2 R)
    w, Rr, kr = rol["contact_width_m"], rol["radius_m"], rol["layer_stiffness_Pa_m"]
    a = (3.0 * force * Rr / (2.0 * w * kr)) ** (1.0 / 3.0)
    contact_len = 2.0 * a
    p_mean = force / (contact_len * w)
    t_dwell = contact_len / speed
    impulse = force / (w * speed)  # integral of p dt through the nip [Pa s]

    # intimate contact (Lee-Springer) and healing during the dwell at T_nip
    aT_nip = arrhenius_shift(T_nip, resin)
    eta_nip = resin["eta_ref_Pas"] * aT_nip
    D_ic = np.minimum(1.0, tack["D0"] * (1.0 + tack["C_ic"] * impulse / eta_nip) ** 0.2)
    D_h = np.minimum(1.0, (t_dwell / (tack["t_heal_ref_s"] * aT_nip)) ** 0.25)

    # cooling after the nip and reduced time for the relaxation of the tack foundation
    tau_c = mat["rho_kg_m3"] * mat["cp_J_kgK"] * mat["thickness_m"] / th["h_tool_W_m2K"]
    t_eval = mdl["evaluation_time_s"]
    tg = np.concatenate([[0.0], np.logspace(-3, np.log10(t_eval), 300)])
    T_hist = th["T_tool_C"] + (T_nip[:, None] - th["T_tool_C"]) * np.exp(-tg[None, :] / tau_c)
    xi = np.trapezoid(1.0 / arrhenius_shift(T_hist, resin), tg, axis=1)
    T_eval = T_hist[:, -1]

    def K_of(reduced_t):
        rel = tack["prony_inf_frac"] + sum(f * np.exp(-reduced_t / tau) for f, tau in tack["prony_terms"])
        return tack_factor * D_ic * tack["k_ref_Pa_m"] * rel

    def Gamma_of(T, rate):
        phi = tack["phi_ref"] * (arrhenius_shift(T, resin) * rate / tack["v_ref_m_s"]) ** tack["phi_exp"]
        return tack_factor * D_ic * D_h * tack["Gamma0_J_m2"] * (1.0 + phi)

    out = {"T_nip_C": T_nip, "T_eval_C": T_eval, "contact_len_m": contact_len,
           "p_mean_Pa": p_mean, "t_dwell_s": t_dwell, "D_ic": D_ic, "D_h": D_h,
           "eps_tension": tension / (E1h * mat["width_m"])}
    for tag, K, G in (("imm", K_of(np.zeros_like(xi)), Gamma_of(T_nip, speed)),
                      ("del", K_of(xi), Gamma_of(T_eval, mdl["delayed_opening_rate_m_s"]))):
        out[f"K_{tag}"] = K
        out[f"Gamma_{tag}"] = G
        out[f"eps_bif_{tag}"] = 2.0 * np.sqrt(D_eff * K) / E1h
        out[f"eps_gamma_{tag}"] = np.sqrt(2.0 * G / E1h)
        out[f"eps_crit_{tag}"] = np.maximum(out[f"eps_bif_{tag}"], out[f"eps_gamma_{tag}"])
        out[f"wavelength_{tag}_m"] = 2.0 * np.pi * (D_eff / K) ** 0.25
    return out


def nip_temperature(speed, proc, params):
    th, mat = params["thermal"], params["material"]
    if "T_nip_C" in proc:
        return proc["T_nip_C"]
    if th["heater"] == "nip_temperature":
        return np.full_like(speed, th["T_nip_C"])
    T_gas = proc.get("T_gas_C", np.full_like(speed, th["T_gas_C"]))
    rc = mat["rho_kg_m3"] * mat["cp_J_kgK"] * mat["thickness_m"]
    expo = th["h_conv_W_m2K"] * th["heated_length_m"] / (rc * speed)
    return T_gas - (T_gas - th["T_initial_C"]) * np.exp(-expo)


# ----------------------------------------------------------------------------- analysis

def analyse(pts, normals, proc_cols, params):
    mdl, prc, mat = params["model"], params["process"], params["material"]
    s, p, nrm, proc = resample(pts, normals, proc_cols, mdl["resample_step_m"])
    y, eps_geo, geo = fibre_strains(s, p, nrm, params)

    n = len(s)
    speed = proc["speed_mm_s"] * 1e-3 if "speed_mm_s" in proc else np.full(n, prc["speed_m_s"])
    force = proc.get("force_N", np.full(n, prc["force_N"]))
    tension = proc.get("tension_N", np.full(n, prc["tension_N"]))
    T_nip = nip_temperature(speed, proc, params)
    factor = tack_field(s, params) * params["tack"]["level"]
    st = interface_state(speed, force, tension, T_nip, params, factor)

    eps = eps_geo + st["eps_tension"][:, None]
    comp = np.maximum(0.0, -eps) + 0.0  # + 0.0 turns -0.0 into 0.0
    r_imm = comp / st["eps_crit_imm"][:, None]
    r_del = comp / st["eps_crit_del"][:, None]
    risk_field = np.maximum(r_imm, r_del)                  # (N, M)
    risk = risk_field.max(axis=1)

    dy = y[1] - y[0]
    governing_del = r_del.max(axis=1) >= r_imm.max(axis=1)
    crit = np.where(governing_del, st["eps_crit_del"], st["eps_crit_imm"])
    bif = np.where(governing_del, st["eps_bif_del"], st["eps_bif_imm"])
    mode = np.where(governing_del, "delayed", "immediate")
    mech = np.where(bif >= crit, "buckling", "peel")

    res = {
        "s_m": s, "x_m": p[:, 0], "y_m": p[:, 1], "z_m": p[:, 2],
        "kappa_g_1_m": geo["kappa_g"], "twist_rad_m": geo["twist"],
        "eps_min": eps.min(axis=1), "eps_max": eps.max(axis=1),
        "risk": risk, "wrinkle": risk >= 1.0,
        "governing_time": mode, "governing_mechanism": mech,
        "wrinkle_width_mm": ((risk_field >= 1.0).sum(axis=1) * dy * 1e3),
        "wavelength_mm": np.where(governing_del, st["wavelength_del_m"], st["wavelength_imm_m"]) * 1e3,
        # minimum in-plane steering radius these process settings can hold (flat tool, no tension)
        "R_crit_m": (mat["width_m"] / 2.0) / np.minimum(st["eps_crit_imm"], st["eps_crit_del"]),
        "speed_m_s": speed, "force_N": force, "tack_factor": factor,
    }
    res.update({k: v for k, v in st.items()})

    E1h = mat["E1_Pa"] * mat["thickness_m"]
    D_eff = mat["E_bend_Pa"] * mat["thickness_m"] ** 3 / 12.0
    lam = res["wavelength_mm"] * 1e-3
    K_gov = np.where(governing_del, st["K_del"], st["K_imm"])

    # spacing: a wrinkle releases the compression; it rebuilds through the tack by shear lag,
    # eps(x) = eps_inf*(1 - exp(-x/l)), so the next wrinkle forms after gap = -l*ln(1 - crit/eps_inf)
    k_s = K_gov * params["tack"]["k_shear_Pa_m"] / params["tack"]["k_ref_Pa_m"]
    l_lag = np.sqrt(E1h / k_s)
    eps_inf = comp.max(axis=1)
    ratio = np.minimum(crit / np.maximum(eps_inf, 1e-300), 1.0 - 1e-12)
    gap = np.where(eps_inf > crit, -l_lag * np.log(1.0 - ratio), np.inf)

    # height (inextensible fibres): each wrinkle stores the length released in its own span
    # (strain relaxed to the clamped-strip buckling strain eps_blister) plus the shear-lag
    # recovery zones on both sides; a raised-cosine bump of length L stores pi^2 H^2 / (4 L)
    eps_blister = 4.0 * np.pi ** 2 * D_eff / (E1h * lam ** 2)
    half_gap = np.where(np.isfinite(gap), gap / 2.0, 0.0)
    released = (lam[:, None] * np.maximum(0.0, comp - eps_blister[:, None])
                + 2.0 * comp * (l_lag * (1.0 - np.exp(-half_gap / l_lag)))[:, None])
    released = np.where(risk_field >= 1.0, released, 0.0)
    height = (2.0 / np.pi) * np.sqrt(lam[:, None] * released)  # (N, M) [m]

    finite_gap = np.where(np.isfinite(gap), gap, 0.0)
    res["wrinkle_height_mm"] = height.max(axis=1) * 1e3
    res["wrinkle_length_mm"] = np.where(res["wrinkle"], lam * 1e3, 0.0)
    res["wrinkle_spacing_mm"] = np.where(res["wrinkle"], (lam + finite_gap) * 1e3, 0.0)
    res["shear_lag_mm"] = l_lag * 1e3

    field = {"y": y, "eps": eps, "risk": risk_field, "points": p, "height": height,
             "wavelength": lam, "wrinkles": layout_wrinkles(s, eps_inf, crit, lam, l_lag, height),
             **geo}
    return res, field


def layout_wrinkles(s, eps_inf, crit, lam, l_lag, height):
    """Individual wrinkles along the path.

    A wrinkle nucleates where the compression reaches the critical strain; it occupies
    [s0, s0 + L]. Behind it the compression rebuilds by shear lag,
    eps(x) = eps_inf(x) * (1 - exp(-(x - end) / l)), and the next wrinkle nucleates at the
    first point where eps(x) >= eps_crit(x).
    """
    out = []
    n = len(s)
    i = int(np.argmax(eps_inf >= crit)) if np.any(eps_inf >= crit) else n
    while i < n:
        end = s[i] + lam[i]
        j = min(int(np.searchsorted(s, end)), n - 1)
        out.append({"s0": s[i], "L": lam[i], "H": height[i:j + 1].max()})
        k = np.arange(j + 1, n)
        ok = eps_inf[k] * (1.0 - np.exp(-(s[k] - end) / l_lag[k])) >= crit[k]
        if not ok.any():
            break
        i = int(k[np.argmax(ok)])
    return out


def write_results(res, path):
    keys = [k for k, v in res.items() if np.ndim(v) == 1]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for i in range(len(res["s_m"])):
            wr.writerow([res[k][i] if isinstance(res[k][i], str) else f"{res[k][i]:.6g}" for k in keys])


def zones(s, flag):
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f or i == len(flag) - 1) and start is not None:
            end = i if f else i - 1
            out.append((s[start], s[end]))
            start = None
    return out


def summary(res, wrinkles):
    s, risk = res["s_m"], res["risk"]
    i = int(np.argmax(risk))
    lines = [
        f"points analysed          : {len(s)}  (length {s[-1]:.3f} m)",
        f"max |in-plane curvature| : {np.abs(res['kappa_g_1_m']).max():.4g} 1/m",
        f"max |twist rate|         : {np.abs(res['twist_rad_m']).max():.4g} rad/m",
        f"max compressive strain   : {-res['eps_min'].min():.3e}",
        f"nip temperature          : {res['T_nip_C'].min():.1f} - {res['T_nip_C'].max():.1f} C",
        f"contact length / dwell   : {res['contact_len_m'].mean()*1e3:.2f} mm / {res['t_dwell_s'].mean():.3f} s",
        f"mean pressure            : {res['p_mean_Pa'].mean()/1e6:.3f} MPa",
        f"intimate contact D_ic    : {res['D_ic'].mean():.3f}   healing D_h: {res['D_h'].mean():.3f}",
        f"tack energy imm / del    : {res['Gamma_imm'].mean():.2f} / {res['Gamma_del'].mean():.2f} J/m2",
        f"tack variability factor  : {res['tack_factor'].min():.2f} - {res['tack_factor'].max():.2f}",
        f"eps_crit imm / del       : {res['eps_crit_imm'].mean():.3e} / {res['eps_crit_del'].mean():.3e}",
        f"min steerable radius     : {np.median(res['R_crit_m']):.3f} m (median), "
        f"{res['R_crit_m'].max():.3f} m (weakest tack)",
        f"max risk index           : {risk[i]:.3f} at s = {s[i]:.3f} m "
        f"({res['governing_time'][i]}, {res['governing_mechanism'][i]}, "
        f"wavelength {res['wavelength_mm'][i]:.1f} mm)",
    ]
    zz = zones(s, res["wrinkle"])
    lines.append(f"wrinkle zones            : {len(zz)}")
    for a, b in zz:
        z = (s >= a) & (s <= b)
        ws = [w for w in wrinkles if a <= w["s0"] <= b]
        if not ws:
            continue
        pitch = np.diff([w["s0"] for w in ws])
        pitch_txt = f"spacing {pitch.mean()*1e3:.0f} mm" if len(pitch) else "single"
        lines.append(f"    s = {a:.3f} - {b:.3f} m : {len(ws)} wrinkle(s), length "
                     f"{np.mean([w['L'] for w in ws])*1e3:.1f} mm, max height "
                     f"{max(w['H'] for w in ws)*1e3:.2f} mm, {pitch_txt}, max width "
                     f"{res['wrinkle_width_mm'][z].max():.1f} mm")
    lines.append(f"total wrinkles           : {len(wrinkles)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- plotting

def wrinkle_polylines(field, res, width_scale, height_scale, step=2.5e-4):
    """Wrinkle profiles as 3D polylines: one per wrinkled fibre across the width.

    Each wrinkle of the layout (start s0, length L) lifts the fibres along the surface
    normal by H(s, y) * (1 - cos(2 pi u / L)) / 2.
    Returns (list of (K, 3) point arrays, list of (point, label) for the zone maxima).
    """
    s = res["s_m"]
    sf = np.arange(0.0, s[-1], step)
    shape = np.zeros_like(sf)
    in_wrinkle = np.zeros(sf.shape, dtype=bool)
    for w in field["wrinkles"]:
        u = (sf >= w["s0"]) & (sf <= w["s0"] + w["L"])
        shape[u] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (sf[u] - w["s0"]) / w["L"]))
        in_wrinkle |= u

    interp3 = lambda a: np.column_stack([np.interp(sf, s, a[:, k]) for k in range(3)])
    C, side, nrm = interp3(field["points"]), interp3(field["side"]), interp3(field["normal"])

    lines = []
    for j, yj in enumerate(field["y"]):
        if field["height"][:, j].max() <= 0.0:
            continue
        Hj = np.interp(sf, s, field["height"][:, j])
        P = C + (yj * width_scale) * side + (Hj * shape * height_scale + 1e-4)[:, None] * nrm
        # keep only the parts where this fibre is wrinkled, split into continuous segments
        idx = np.flatnonzero(in_wrinkle & (Hj > 0.0))
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) > 1:
                lines.append(P[seg])

    label_pts = []
    for a, b in zones(s, res["wrinkle"]):
        z = (s >= a) & (s <= b)
        i = np.flatnonzero(z)[np.argmax(res["wrinkle_height_mm"][z])]
        j = int(np.argmax(field["height"][i]))
        pt = (field["points"][i] + field["y"][j] * width_scale * field["side"][i]
              + field["height"][i, j] * height_scale * field["normal"][i])
        label_pts.append((pt, f"H {res['wrinkle_height_mm'][i]:.2f} mm, L {res['wrinkle_length_mm'][i]:.1f} mm"))
    return lines, label_pts


def plot(field, res, params, title, width_scale=10.0, height_scale=20.0, screenshot=None):
    import pyvista as pv
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    y = field["y"] * width_scale
    P = field["points"][:, None, :] + y[None, :, None] * field["side"][:, None, :]
    grid = pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2])
    grid.point_data["risk"] = field["risk"].ravel(order="F")

    # 0-1: green -> yellow -> orange (safe, rising risk); >= 1: red (wrinkle)
    safe = plt.get_cmap("RdYlGn_r")(np.linspace(0.0, 0.75, 128))
    red = np.tile([0.85, 0.0, 0.0, 1.0], (128, 1))
    cmap = ListedColormap(np.vstack([safe, red]))

    pl = pv.Plotter(off_screen=screenshot is not None, window_size=(1400, 800))
    pl.add_mesh(grid, scalars="risk", cmap=cmap, clim=[0.0, 2.0], show_edges=False,
                scalar_bar_args={"title": "Wrinkle risk (>=1 red = wrinkle)"})
    pl.add_mesh(pv.lines_from_points(field["points"]), color="black", line_width=2)

    lines, labels = wrinkle_polylines(field, res, width_scale, height_scale)
    for pts in lines:
        pl.add_mesh(pv.lines_from_points(pts), color="black", line_width=1.5)
    if labels:
        pl.add_point_labels(np.array([p for p, _ in labels]), [t for _, t in labels],
                            font_size=12, point_size=8, point_color="black", always_visible=True)

    pl.add_text(f"{title}\nwidth x{width_scale:g}, wrinkle height x{height_scale:g} for display | "
                f"max risk {res['risk'].max():.2f} | wrinkles {len(field['wrinkles'])} | "
                f"max height {res['wrinkle_height_mm'].max():.2f} mm", font_size=10)
    pl.show_axes()
    pl.view_isometric()
    if screenshot:
        pl.screenshot(screenshot)
        pl.close()
    else:
        pl.show()


# ----------------------------------------------------------------------------- cases

TITLE = "Tape Placement - Wrinkles defects prediction model (physics-based)"


def build_case(name):
    """Returns (points, normals, per-point process columns, params, title) of a built-in case."""
    params = copy.deepcopy(PARAMS)
    if name == "s_course":
        pts, normals, proc = demo_path(DEMO_FLAT_M, DEMO_ARC1_RADIUS_M, DEMO_ARC1_ANGLE_DEG,
                                       DEMO_ARC2_R_START_M, DEMO_ARC2_R_END_M, DEMO_ARC2_LENGTH_M)
        title = (f"{TITLE}: {DEMO_FLAT_M:g} m straight, R {DEMO_ARC1_RADIUS_M:g} m x "
                 f"{DEMO_ARC1_ANGLE_DEG:g} deg, then opposite R {DEMO_ARC2_R_START_M:g} -> "
                 f"{DEMO_ARC2_R_END_M:g} m over {DEMO_ARC2_LENGTH_M:g} m")
    elif name == "arc_1500":
        pts, normals, proc = arc_path(ARC_RADIUS_M, ARC_ANGLE_DEG)
        params["material"]["width_m"] = ARC_WIDTH_M
        params["material"]["thickness_m"] = ARC_THICKNESS_M
        title = (f"{TITLE}: constant R {ARC_RADIUS_M*1e3:g} mm x {ARC_ANGLE_DEG:g} deg, tape "
                 f"{ARC_WIDTH_M*1e3:g} x {ARC_THICKNESS_M*1e3:g} mm, tack CoV {ARC_TACK_COV:g}")
        if ARC_TARGET_R_CRIT_M:
            # R_crit ~ 1/eps_crit ~ 1/sqrt(tack level): level = (R_median / R_target)^2
            res, _ = analyse(pts, normals, proc, params)
            params["tack"]["level"] = (np.median(res["R_crit_m"]) / ARC_TARGET_R_CRIT_M) ** 2
            title += f", tack level {params['tack']['level']:.2f} (calibrated R_crit {ARC_TARGET_R_CRIT_M:g} m)"
        params["variability"].update(cov=ARC_TACK_COV, correlation_length_m=ARC_TACK_CORR_LENGTH_M,
                                     seed=ARC_TACK_SEED)
    else:
        sys.exit(f"unknown case: {name}")
    return pts, normals, proc, params, title


def run_case(pts, normals, proc, params, title):
    res, field = analyse(pts, normals, proc, params)
    write_results(res, OUT_CSV)
    print(title)
    print(summary(res, field["wrinkles"]))
    print(f"results written to {OUT_CSV}")
    if SHOW_PLOT or SCREENSHOT:
        plot(field, res, params, title, WIDTH_SCALE, HEIGHT_SCALE, SCREENSHOT)


def sensitivity_study(case="arc_1500"):
    """Heating (hot gas temperature) x deposition speed (compaction dwell time) on one case.

    The tack variability field is the same in every run (same seed), so differences
    come from the process only.
    """
    pts, normals, proc, base, title = build_case(case)
    rows = []
    for v in SENS_SPEEDS_MM_S:
        for Tg in SENS_T_GAS_C:
            params = copy.deepcopy(base)
            params["process"]["speed_m_s"] = v * 1e-3
            params["thermal"]["heater"] = "hot_gas"
            params["thermal"]["T_gas_C"] = Tg
            res, field = analyse(pts, normals, proc, params)
            ws = field["wrinkles"]
            rows.append({
                "speed_mm_s": v, "T_gas_C": Tg,
                "T_nip_C": res["T_nip_C"].mean(), "t_dwell_s": res["t_dwell_s"].mean(),
                "D_ic": res["D_ic"].mean(), "D_h": res["D_h"].mean(),
                "Gamma_imm_J_m2": res["Gamma_imm"].mean(), "Gamma_del_J_m2": res["Gamma_del"].mean(),
                "R_crit_median_m": np.median(res["R_crit_m"]),
                "at_risk_length_pct": 100.0 * res["wrinkle"].mean(),
                "n_wrinkles": len(ws),
                "max_height_mm": max((w["H"] for w in ws), default=0.0) * 1e3,
            })

    keys = list(rows[0])
    with open(SENS_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            wr.writerow([f"{r[k]:.6g}" for k in keys])
    print(f"sensitivity study on: {title}")
    print(f"{'speed':>6} {'T_gas':>6} {'T_nip':>6} {'dwell':>6} {'D_ic':>5} {'D_h':>5} "
          f"{'Gam_imm':>7} {'R_crit':>7} {'wrinkles':>8} {'H_max':>6}")
    for r in rows:
        print(f"{r['speed_mm_s']:6.0f} {r['T_gas_C']:6.0f} {r['T_nip_C']:6.1f} {r['t_dwell_s']:6.3f} "
              f"{r['D_ic']:5.2f} {r['D_h']:5.2f} {r['Gamma_imm_J_m2']:7.2f} {r['R_crit_median_m']:7.3f} "
              f"{r['n_wrinkles']:8d} {r['max_height_mm']:6.2f}")
    print(f"results written to {SENS_CSV}")
    plot_sensitivity(rows, title.split(": ", 1)[-1])


def plot_sensitivity(rows, title):
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid, axis, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    blue = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]    # ordinal: slow -> fast
    orange = ["#f39a70", "#eb6834", "#c9521f", "#9e3e14", "#72290a"]  # ordinal: cold -> hot
    pick = lambda ramp, n: [ramp[round(i * (len(ramp) - 1) / max(n - 1, 1))] for i in range(n)]

    speeds = SENS_SPEEDS_MM_S
    # at most 5 heating levels as series, evenly spread over the studied range
    temps = [SENS_T_GAS_C[round(i * (len(SENS_T_GAS_C) - 1) / 4)] for i in range(5)] \
        if len(SENS_T_GAS_C) > 5 else SENS_T_GAS_C
    dwell = {v: next(r["t_dwell_s"] for r in rows if r["speed_mm_s"] == v) for v in speeds}

    metrics = [("T_nip_C", "Nip temperature (°C)"),
               ("R_crit_median_m", "Min. steerable radius, median (m)"),
               ("n_wrinkles", "Number of wrinkles"),
               ("at_risk_length_pct", "Length at risk: risk >= 1 (% of path)")]
    # row 0: x = heating, one line per speed (no nip-temperature panel: its slot holds the legend)
    # row 1: x = speed, one line per heating level
    rows_cfg = [
        ("T_gas_C", "Hot gas temperature (°C)", "speed_mm_s", speeds, pick(blue, len(speeds)),
         lambda v: f"{v:g} mm/s (dwell {dwell[v]*1e3:.0f} ms)", "Deposition speed", {"T_nip_C"}),
        ("speed_mm_s", "Deposition speed (mm/s) / compaction dwell time", "T_gas_C", temps,
         pick(orange, len(temps)), lambda T: f"gas {T:g} °C", "Hot gas temperature", set()),
    ]

    fig, axes = plt.subplots(2, len(metrics), figsize=(21, 9.5), facecolor=surface)
    for r_i, (xkey, xlabel, skey, series, colors, fmt, legend_title, skip) in enumerate(rows_cfg):
        for c_i, (key, label) in enumerate(metrics):
            ax = axes[r_i, c_i]
            if key in skip:
                ax.axis("off")
                continue
            ax.set_facecolor(surface)
            for val, col in zip(series, colors):
                rr = sorted((r for r in rows if r[skey] == val), key=lambda r: r[xkey])
                ax.plot([r[xkey] for r in rr], [r[key] for r in rr], color=col, lw=2,
                        marker="o", ms=5, mec=surface, mew=1.5, label=fmt(val))
            if key == "R_crit_median_m":
                ax.axhline(ARC_RADIUS_M, color=muted, lw=1, ls="--")
                ax.annotate(f"path radius {ARC_RADIUS_M:g} m", (1.0, ARC_RADIUS_M),
                            xycoords=("axes fraction", "data"), xytext=(0, 4),
                            textcoords="offset points", ha="right", color=ink2, fontsize=9)
            if key == "n_wrinkles":
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            if xkey == "speed_mm_s":
                ax.set_xticks(speeds)
                ax.set_xticklabels([f"{v:g}\n{dwell[v]*1e3:.0f} ms" for v in speeds])
            ax.set_title(label, loc="left", color=ink, fontsize=11)
            ax.set_xlabel(xlabel, color=ink2, fontsize=9)
            ax.grid(True, color=grid, lw=0.6)
            ax.tick_params(colors=muted, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(axis)
        handles, labels = axes[r_i, -1].get_legend_handles_labels()
        free = [c for c, (key, _) in enumerate(metrics) if key in skip]
        if free:  # legend in the emptied panel
            axes[r_i, free[0]].legend(handles, labels, loc="center", frameon=False, fontsize=10,
                                      labelcolor=ink2, title=legend_title, title_fontsize=10)
        else:
            axes[r_i, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9,
                                 labelcolor=ink2, title=legend_title, title_fontsize=9, alignment="left")
    fig.suptitle(f"Wrinkle sensitivity to heating and deposition speed\n{title}",
                 color=ink, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(SENS_PNG, dpi=150, facecolor=surface)
    print(f"figure written to {SENS_PNG}")
    if SHOW_PLOT and not SCREENSHOT:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():

    sensitivity_study("arc_1500")

    run_case(*build_case("arc_1500"))
    
    # pts, normals, proc = read_path_csv(PATH_CSV, PATH_UNITS)
    # run_case(pts, normals, proc, PARAMS, f"{TITLE}: {PATH_CSV}")

    # run_case(*build_case("s_course"))

if __name__ == "__main__":
    main()
