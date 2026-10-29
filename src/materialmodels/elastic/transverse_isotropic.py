"""Transversely isotropic (fiber-reinforced) elastic model. Voigt order 11, 22, 33, 12, 13, 23."""

import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.tensors import (
    rotate_tensor4,
    rotate_tensor4_field,
    rotation_field_from_directions,
    rotation_from_direction,
    tensor4_to_voigt,
    voigt_to_tensor4,
)


class TransverseIsotropic(ConstitutiveModel):
    """
    Transversely isotropic material with the reference fiber axis Z = [0, 0, 1].

    ``elastic_stiffness_tensor()`` is rotated into ``fiber_dir``.
    ``stiffness_tensor_rotated`` and ``stiffness_field_oriented`` rotate the
    reference tensor to an explicit direction instead (they override
    ``fiber_dir``, not compose with it).

    Parameters
    ----------
    E_L, E_T    : Young's moduli along and across the fiber
    G_LT        : shear modulus in planes containing the fiber
    nu_LT       : Poisson's ratio, -eps_T / eps_L under load along L
    nu_TT, G_TT : transverse-plane constants; give exactly one, the other
                  follows from G_TT = E_T / (2 (1 + nu_TT)); nu_TT outside
                  (-1, 1) raises
    fiber_dir   : (3,) direction (default Z), or (3, Nv) per-voxel field over
                  the whole grid, which makes elastic_stiffness_tensor()
                  a (3, 3, 3, 3, Nv) field
    name        : display label
    k_res, Gc   : phase-field parameters (see LinearElasticIsotropic); fracture
                  uses an isotropized proxy of this material
    """

    def __init__(
        self,
        E_L: float,
        E_T: float,
        G_LT: float,
        nu_LT: float,
        nu_TT: float | None = None,
        G_TT: float | None = None,
        fiber_dir: jnp.ndarray | None = None,
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
        self.fiber_dir = (jnp.array([0.0, 0.0, 1.0]) if fiber_dir is None
                           else jnp.asarray(fiber_dir, dtype=float))
        self.name = name
        self.k_res = float(k_res)
        self.Gc = float(Gc) if Gc is not None else None

        if G_TT is not None:
            self.G_TT = float(G_TT)
            self.nu_TT = float(self.E_T / (2.0 * self.G_TT) - 1.0)
        else:
            assert nu_TT is not None
            self.nu_TT = float(nu_TT)
            self.G_TT = float(self.E_T / (2.0 * (1.0 + self.nu_TT)))

        # A 2-D isotropic plane allows -1 < nu < 1, not the 3-D bound 0.5.
        if not (-1.0 < self.nu_TT < 1.0):
            raise ValueError(
                f"TransverseIsotropic: nu_TT={self.nu_TT:.4g} (E_T={self.E_T:.4g}, "
                f"G_TT={self.G_TT:.4g}) is outside the physically valid range "
                f"(-1, 1) for an isotropic transverse plane -- check E_T/G_TT/nu_TT "
                f"are mutually consistent."
            )

        self.nu_TL = float(nu_LT * E_T / E_L)

    def _compliance_engineering(self) -> jnp.ndarray:
        """(6, 6) engineering compliance, fiber along Z (Voigt index 2)."""
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

    def _stiffness_tensor_reference(self) -> jnp.ndarray:
        """(3, 3, 3, 3) stiffness in the reference frame (fiber along Z)."""
        C_eng = jnp.linalg.inv(self._compliance_engineering())
        return jnp.array(voigt_to_tensor4(C_eng, engineering=True))

    def elastic_stiffness_tensor(self) -> jnp.ndarray:
        """Stiffness rotated into ``fiber_dir``: (3, 3, 3, 3), or (3, 3, 3, 3, Nv) for a per-voxel field."""
        if self.fiber_dir.ndim > 1:
            return self.stiffness_field_oriented(self.fiber_dir)
        R = rotation_from_direction(self.fiber_dir)
        return rotate_tensor4(R, self._stiffness_tensor_reference())

    def stiffness_tensor_rotated(self, fiber_dir: jnp.ndarray) -> jnp.ndarray:
        """(3, 3, 3, 3) reference stiffness rotated so the fiber points along ``fiber_dir``."""
        R = rotation_from_direction(jnp.asarray(fiber_dir, float))
        return rotate_tensor4(R, self._stiffness_tensor_reference())

    def stiffness_field_oriented(
        self, orientations: jnp.ndarray, C_ref: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Reference stiffness rotated per voxel.

        Parameters
        ----------
        orientations : (3, Nv) fiber direction per voxel
        C_ref        : reference tensor; pass it precomputed when calling
                       inside a jit/pmap trace, because building it uses numpy
                       (e.g. ``stiffness_tensor_rotated([0, 0, 1])``)

        Returns
        -------
        C_field : (3, 3, 3, 3, Nv)
        """
        if C_ref is None:
            C_ref = self._stiffness_tensor_reference()
        R_field = rotation_field_from_directions(orientations)
        return rotate_tensor4_field(R_field, C_ref)

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """(6, 6) Voigt form of ``elastic_stiffness_tensor()``; ``engineering`` as LinearElasticIsotropic."""
        return jnp.array(tensor4_to_voigt(self.elastic_stiffness_tensor(), engineering=engineering))  # type: ignore[arg-type]

    def stress_voigt(self, eps_voigt: jnp.ndarray, engineering: bool = False) -> jnp.ndarray:
        """Voigt stress from Voigt strain, (..., 6) -> (..., 6)."""
        return eps_voigt @ self.stiffness_voigt(engineering=engineering).T

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"TransverseIsotropic{tag}: "
                f"E_L={self.E_L:.3g}  E_T={self.E_T:.3g}  G_LT={self.G_LT:.3g}  "
                f"G_TT={self.G_TT:.3g}  nu_LT={self.nu_LT:.3g}  nu_TT={self.nu_TT:.3g}")
