"""
Transversely isotropic (fibre-reinforced) elastic material model.

Voigt index convention (Abaqus order):
    0 = 11,  1 = 22,  2 = 33,  3 = 12,  4 = 13,  5 = 23

Rebuilt on materialmodels.tensors -- the old (pre-refactor) TransverseIsotropicFibre
hand-rolled its own private Voigt-conversion/rotation helpers; this version calls
the shared, general versions of those same operations instead.
"""

import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.tensors import (
    rotate_tensor4,
    rotation_from_direction,
    tensor4_to_voigt,
    voigt_to_tensor4,
)


class TransverseIsotropic(ConstitutiveModel):
    """
    Transversely isotropic (fibre-reinforced) material.

    The *reference* fibre axis is local  **Z = [0, 0, 1]**.
    Call ``stiffness_tensor_rotated(fiber_dir)`` to obtain the stiffness
    for an arbitrary fibre direction in the global frame.

    5 independent elastic constants (engineering / test-data notation):

    E_L   : Young's modulus along the fibre (longitudinal)
    E_T   : Young's modulus perpendicular to the fibre (transverse)
    G_LT  : Shear modulus in planes that contain the fibre axis
    nu_LT : Poisson's ratio  (-eps_T / eps_L when loaded uniaxially along L)
    nu_TT, G_TT : the transverse-transverse plane's Poisson's ratio and
                shear modulus are *not* independent (isotropic plane:
                G_TT = E_T / (2*(1 + nu_TT))) -- give exactly one, the
                other is derived. G_TT is often the one actually reported
                in test data/literature, so it's accepted directly instead
                of forcing a manual nu_TT = E_T/(2*G_TT) - 1 conversion at
                the call site. Either way, the resulting nu_TT is checked
                against the physically valid range for a 2-D isotropic
                plane (-1, 1) -- an inconsistent E_T/G_TT pair raises
                immediately rather than silently building a nonphysical
                material.

    Derived:
    nu_TL = nu_LT * E_T / E_L       reciprocal Poisson ratio (symmetry of S)

    k_res, Gc : AT2 phase-field residual stiffness / critical energy release
                rate -- see materialmodels.elastic.isotropic.
                LinearElasticIsotropic's docstring for what these do; same
                meaning here. Only read by fracture problems, which use an
                isotropized (λ, μ) proxy of this material's stiffness_tensor()
                for the Amor driving-force split -- see materialmodels.
                phasefield.driving_force.lame_field's docstring for why
                that's an approximation for an anisotropic material like
                this one, not new plumbing.
    """

    def __init__(
        self,
        E_L: float,
        E_T: float,
        G_LT: float,
        nu_LT: float,
        nu_TT: float | None = None,
        G_TT: float | None = None,
        name: str = "",
        k_res: float = 1e-6,
        Gc: float | None = None,
    ):
        if (nu_TT is None) == (G_TT is None):
            raise ValueError(
                "TransverseIsotropic: specify exactly one of nu_TT or G_TT -- "
                "they're not independent for the isotropic transverse plane "
                "(G_TT = E_T / (2*(1 + nu_TT))), so giving both (or neither) "
                "is ambiguous."
            )

        self.E_L = float(E_L)
        self.E_T = float(E_T)
        self.G_LT = float(G_LT)
        self.nu_LT = float(nu_LT)
        self.name = name
        self.k_res = float(k_res)
        self.Gc = float(Gc) if Gc is not None else None

        if G_TT is not None:
            self.G_TT = float(G_TT)
            self.nu_TT = float(self.E_T / (2.0 * self.G_TT) - 1.0)
        else:
            assert nu_TT is not None  # narrows for type checkers; guaranteed by the xor check above
            self.nu_TT = float(nu_TT)
            self.G_TT = float(self.E_T / (2.0 * (1.0 + self.nu_TT)))

        # double check: nu_TT (given directly, or derived from G_TT) must be
        # physically valid for a 2-D isotropic plane -- unlike the usual 3-D
        # bound of 0.5, a plane-stress/strain isotropic sheet's Poisson's
        # ratio range is (-1, 1); outside that, G_TT and E_T were given an
        # inconsistent pair.
        if not (-1.0 < self.nu_TT < 1.0):
            raise ValueError(
                f"TransverseIsotropic: nu_TT={self.nu_TT:.4g} (E_T={self.E_T:.4g}, "
                f"G_TT={self.G_TT:.4g}) is outside the physically valid range "
                f"(-1, 1) for an isotropic transverse plane -- check E_T/G_TT/nu_TT "
                f"are mutually consistent."
            )

        self.nu_TL = float(nu_LT * E_T / E_L)   # S symmetry: nu_LT/E_L = nu_TL/E_T

    # ------------------------------------------------------------------
    # Stiffness representations
    # ------------------------------------------------------------------

    def _compliance_engineering(self) -> jnp.ndarray:
        """6x6 engineering compliance S (fibre along Z = Voigt index 2)."""
        s_tt = -self.nu_TT / self.E_T
        s_lt = -self.nu_LT / self.E_L
        return jnp.array([
            [1.0 / self.E_T, s_tt,           s_lt,           0.0,             0.0,             0.0],
            [s_tt,           1.0 / self.E_T, s_lt,           0.0,             0.0,             0.0],
            [s_lt,           s_lt,           1.0 / self.E_L, 0.0,             0.0,             0.0],
            [0.0,            0.0,            0.0,            1.0 / self.G_TT, 0.0,             0.0],
            [0.0,            0.0,            0.0,            0.0,             1.0 / self.G_LT, 0.0],
            [0.0,            0.0,            0.0,            0.0,             0.0,             1.0 / self.G_LT],
        ])

    def stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) stiffness in the reference frame (fibre along Z)."""
        C_eng = jnp.linalg.inv(self._compliance_engineering())
        return jnp.array(voigt_to_tensor4(C_eng, engineering=True))

    def stiffness_tensor_rotated(self, fiber_dir: jnp.ndarray) -> jnp.ndarray:
        """
        (3, 3, 3, 3) stiffness rotated so the fibre axis aligns with
        ``fiber_dir`` in the global frame.

        Parameters
        ----------
        fiber_dir : (3,) unit vector (global frame)
        """
        R = rotation_from_direction(jnp.asarray(fiber_dir, float))
        return rotate_tensor4(R, self.stiffness_tensor())

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """
        6x6 Voigt stiffness in the reference frame (fibre along Z).

        engineering=False (default) -> tensor shear convention (compatible
        with ``post.fields.to_voigt`` and the FFT solver).
        engineering=True            -> Abaqus / UMAT gamma convention.
        """
        # tensor4_to_voigt's own param type is np.ndarray (materialmodels.tensors
        # is deliberately plain numpy -- one-time setup, not per-voxel; see its
        # module docstring), but it np.asarray()s its input internally regardless,
        # so a jax array works fine at runtime despite the nominal type mismatch.
        return jnp.array(tensor4_to_voigt(self.stiffness_tensor(), engineering=engineering))  # type: ignore[arg-type]

    def stress_voigt(self, eps_voigt: jnp.ndarray, engineering: bool = False) -> jnp.ndarray:
        """Compute Voigt stress from Voigt strain (..., 6) -> (..., 6)."""
        return eps_voigt @ self.stiffness_voigt(engineering=engineering).T

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"TransverseIsotropic{tag}: "
                f"E_L={self.E_L:.3g}  E_T={self.E_T:.3g}  G_LT={self.G_LT:.3g}  "
                f"G_TT={self.G_TT:.3g}  nu_LT={self.nu_LT:.3g}  nu_TT={self.nu_TT:.3g}")
