"""
Micromechanics helpers for computing homogenised yarn (tow) properties
from constituent fibre and matrix data plus the fibre volume fraction.

Models
------
Longitudinal  (E_L, nu_LT) : rule of mixtures
Transverse    (E_T)         : Halpin-Tsai,  ξ = 2  (circular fibre cross-section)
Long. shear   (G_LT)        : Halpin-Tsai,  ξ = 1
Trans. shear  (G_TT)        : Halpin-Tsai,  ξ = 1
Transv. Poisson (nu_TT)     : derived from E_T and G_TT via isotropy relation
"""


def halpin_tsai(p_f: float, p_m: float, Vf: float, xi: float) -> float:
    """
    Halpin-Tsai mixing rule for a single modulus.

        η = (p_f/p_m − 1) / (p_f/p_m + ξ)
        P = p_m · (1 + ξ·η·Vf) / (1 − η·Vf)

    Parameters
    ----------
    p_f : fibre property
    p_m : matrix property
    Vf  : fibre volume fraction  ∈ (0, 1)
    xi  : reinforcing factor (2 for E_T with round fibre, 1 for G)
    """
    eta = (p_f / p_m - 1.0) / (p_f / p_m + xi)
    return p_m * (1.0 + xi * eta * Vf) / (1.0 - eta * Vf)


def yarn_properties(
    Vf:     float,
    E_fL:   float,
    E_fT:   float,
    G_fLT:  float,
    nu_fLT: float,
    nu_fTT: float,
    E_m:    float,
    nu_m:   float,
) -> tuple[float, float, float, float, float]:
    """
    Homogenised transverse-isotropic yarn properties from constituents.

    Parameters
    ----------
    Vf     : fibre volume fraction within the yarn/tow  ∈ (0, 1)
    E_fL   : fibre longitudinal Young's modulus
    E_fT   : fibre transverse Young's modulus
    G_fLT  : fibre longitudinal-transverse shear modulus
    nu_fLT : fibre longitudinal-transverse Poisson ratio
    nu_fTT : fibre transverse-transverse Poisson ratio
    E_m    : matrix Young's modulus
    nu_m   : matrix Poisson ratio
    (all moduli in the same unit, e.g. MPa)

    Returns
    -------
    E_L, E_T, G_LT, nu_LT, nu_TT
        Five independent constants of the homogenised transverse-isotropic yarn.
        Use them directly with ``TransverseIsotropicFibre``.
    """
    G_m   = E_m   / (2.0 * (1.0 + nu_m))
    G_fTT = E_fT  / (2.0 * (1.0 + nu_fTT))

    # longitudinal — rule of mixtures
    E_L   = Vf * E_fL   + (1.0 - Vf) * E_m
    nu_LT = Vf * nu_fLT + (1.0 - Vf) * nu_m

    # transverse / shear — Halpin-Tsai
    E_T  = halpin_tsai(E_fT,  E_m,  Vf, xi=2.0)
    G_LT = halpin_tsai(G_fLT, G_m,  Vf, xi=1.0)
    G_TT = halpin_tsai(G_fTT, G_m,  Vf, xi=1.0)

    nu_TT = E_T / (2.0 * G_TT) - 1.0

    return E_L, E_T, G_LT, nu_LT, nu_TT
