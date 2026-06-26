"""
Semi-empirical micromechanics for transverse-isotropic yarn/tow properties.

Formulae from:
    Schürmann, H.: Konstruieren mit Faser-Kunststoff-Verbunden.
    Springer, Berlin Heidelberg, 2005.

All moduli must be in the same unit (e.g. MPa).

Notation
--------
  ∥  = fibre-parallel direction  (1-direction)
  ⊥  = fibre-transverse direction (2 = 3 for transverse isotropy)
  φ  = fibre volume fraction within the yarn/tow
"""


# ── Building-block formulas ───────────────────────────────────────────────────

def _E_L(Vf: float, E_fL: float, E_m: float) -> float:
    """E∥  — rule of mixtures."""
    return Vf * E_fL + (1.0 - Vf) * E_m


def _E_T(Vf: float, E_fT: float, E_m: float, nu_m: float) -> float:
    """
    E⊥  — semi-empirical Halpin-Tsai with plane-strain matrix modulus.

        E⊥ = E*_m · (1 + 0.85 φ²) / [(1−φ)^1.25 + (E*_m / E_f⊥) · φ]

    where  E*_m = E_m / (1 − ν_m²)  is the plane-strain (constrained) modulus.
    """
    Em_star = E_m / (1.0 - nu_m ** 2)
    return Em_star * (1.0 + 0.85 * Vf ** 2) / (
        (1.0 - Vf) ** 1.25 + (Em_star / E_fT) * Vf
    )


def _G_LT(Vf: float, G_fLT: float, G_m: float) -> float:
    """
    G∥⊥  — semi-empirical shear modulus.

        G∥⊥ = G_m · (1 + 0.4 φ^0.5) / [(1−φ)^1.45 + (G_m / G_f∥⊥) · φ]
    """
    return G_m * (1.0 + 0.4 * Vf ** 0.5) / (
        (1.0 - Vf) ** 1.45 + (G_m / G_fLT) * Vf
    )


def _nu_LT(Vf: float, nu_fLT: float, nu_m: float) -> float:
    """ν∥⊥  — rule of mixtures (major Poisson ratio)."""
    return Vf * nu_fLT + (1.0 - Vf) * nu_m


def _nu_m_eff(E_L: float, nu_LT: float, E_m: float, nu_m: float) -> float:
    """
    Effective matrix Poisson ratio accounting for fibre constraint (Schürmann).

        ν_m,eff = ν_m · (1 + ν_m − ν∥⊥ · E_m/E∥) / (1 − ν_m² + ν_m · ν∥⊥ · E_m/E∥)
    """
    r = nu_LT * E_m / E_L
    return nu_m * (1.0 + nu_m - r) / (1.0 - nu_m ** 2 + nu_m * r)


def _nu_TT(
    Vf: float,
    nu_fTT: float,
    E_m: float, nu_m: float,
    E_L: float, nu_LT: float,
) -> float:
    """
    ν⊥⊥  — rule of mixtures with effective matrix Poisson ratio.

        ν⊥⊥ = φ · ν_f + (1−φ) · ν_m,eff
    """
    nu_meff = _nu_m_eff(E_L, nu_LT, E_m, nu_m)
    return Vf * nu_fTT + (1.0 - Vf) * nu_meff


# ── Public API ────────────────────────────────────────────────────────────────

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
    Homogenised transverse-isotropic yarn (tow) properties from constituents.

    Parameters
    ----------
    Vf     : fibre volume fraction within the yarn/tow  ∈ (0, 1)
    E_fL   : fibre longitudinal Young's modulus  (∥)
    E_fT   : fibre transverse Young's modulus    (⊥)
    G_fLT  : fibre ∥⊥ shear modulus
    nu_fLT : fibre major Poisson ratio  ν_f∥⊥
    nu_fTT : fibre transverse Poisson ratio  ν_f⊥⊥
    E_m    : matrix Young's modulus
    nu_m   : matrix Poisson ratio
    (all moduli in the same unit, e.g. MPa)

    Returns
    -------
    E_L, E_T, G_LT, nu_LT, nu_TT
        Five independent constants of the homogenised transverse-isotropic yarn.
        Pass directly to ``TransverseIsotropicFibre``;
        G_TT = E_T / (2·(1 + nu_TT)) is derived there from isotropy.

    Reference
    ---------
    Schürmann (2005), Konstruieren mit FKV, Springer.
    """
    G_m   = E_m / (2.0 * (1.0 + nu_m))

    E_L   = _E_L(Vf, E_fL, E_m)
    E_T   = _E_T(Vf, E_fT, E_m, nu_m)
    G_LT  = _G_LT(Vf, G_fLT, G_m)
    nu_LT = _nu_LT(Vf, nu_fLT, nu_m)
    nu_TT = _nu_TT(Vf, nu_fTT, E_m, nu_m, E_L, nu_LT)

    return E_L, E_T, G_LT, nu_LT, nu_TT
