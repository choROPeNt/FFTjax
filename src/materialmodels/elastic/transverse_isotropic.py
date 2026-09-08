"""
Transversely isotropic (fibre-reinforced) elastic material model.

Voigt index convention (Abaqus order):
    0 = 11,  1 = 22,  2 = 33,  3 = 12,  4 = 13,  5 = 23

Rebuilt on materialmodels.tensors -- the old (pre-refactor) TransverseIsotropicFibre
hand-rolled its own private Voigt-conversion/rotation helpers; this version calls
the shared, general versions of those same operations instead.
"""

import jax
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

    The *reference* fibre axis is local **Z = [0, 0, 1]**. ``fiber_dir``
    (constructor arg, default Z i.e. no rotation) is this material's own
    fixed orientation in the global frame -- ``stiffness_tensor()`` always
    returns the tensor already rotated into it, so
    ``materialmodels.assembly.assemble_C_field`` (or any other caller that
    just calls ``.stiffness_tensor()``) gets a correctly oriented per-phase
    stiffness automatically, with no special-case rotation plumbing needed
    at the assembly layer. ``stiffness_tensor_rotated(fiber_dir)`` and
    ``stiffness_field_oriented(orientations)`` are a different pair of
    functions: they rotate the *reference* tensor by an explicitly given
    direction (one, or a per-voxel field), overriding this material's own
    stored ``fiber_dir`` rather than composing with it -- for a one-off
    direction, or a spatially varying orientation field, independent of
    whatever this material's default is.

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

    fiber_dir : (3,) unit vector (global frame), default None -> Z = [0, 0, 1]
                (identity rotation, i.e. stiffness_tensor() then returns
                exactly the reference-frame tensor, same as before this
                parameter existed). Not required to be pre-normalized --
                rotation_from_direction normalizes internally.

                Also accepts a per-voxel field, (3, Nv) -- one direction per
                voxel, spanning the WHOLE grid (e.g. straight from
                utils.io.reader.read_vtu's orientations output), not just
                this material's own phase -- in which case stiffness_tensor()
                returns the (3,3,3,3,Nv) rotated FIELD instead of a single
                (3,3,3,3) tensor (vmapped internally, same machinery as
                stiffness_field_oriented). materialmodels.assembly.
                assemble_C_field detects this automatically and mixes it
                correctly with other, constant-tensor materials in the same
                materials: list -- see its own docstring.

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

    def _stiffness_tensor_reference(self) -> jnp.ndarray:
        """(3, 3, 3, 3) stiffness in the reference frame (fibre along Z),
        before this material's own ``fiber_dir`` is applied. Internal --
        ``stiffness_tensor_rotated``/``stiffness_field_oriented`` build on
        this directly (an explicitly given direction overrides, rather than
        composes with, ``fiber_dir``); external callers wanting the
        reference tensor itself can still get it via
        ``stiffness_tensor_rotated([0, 0, 1])``."""
        C_eng = jnp.linalg.inv(self._compliance_engineering())
        return jnp.array(voigt_to_tensor4(C_eng, engineering=True))

    def stiffness_tensor(self) -> jnp.ndarray:
        """
        Stiffness rotated into this material's own ``fiber_dir`` (default Z,
        i.e. identity rotation) -- the ConstitutiveModel-required
        no-argument method, so ``materialmodels.assembly.assemble_C_field``
        (or any other caller that just calls ``.stiffness_tensor()``) gets
        the correctly oriented tensor automatically for a fixed per-phase
        fibre direction.

        Shape follows ``fiber_dir``: ``(3, 3, 3, 3)`` for a single direction
        (the common case), or ``(3, 3, 3, 3, Nv)`` if ``fiber_dir`` was
        given as a per-voxel field -- ``assemble_C_field`` handles both
        shapes automatically, including mixed with other materials.
        """
        if self.fiber_dir.ndim > 1:
            return self.stiffness_field_oriented(self.fiber_dir)
        R = rotation_from_direction(self.fiber_dir)
        return rotate_tensor4(R, self._stiffness_tensor_reference())

    def stiffness_tensor_rotated(self, fiber_dir: jnp.ndarray) -> jnp.ndarray:
        """
        (3, 3, 3, 3) stiffness rotated so the fibre axis aligns with an
        explicitly given ``fiber_dir`` in the global frame -- overrides this
        material's own stored ``fiber_dir`` rather than composing with it
        (rotates the *reference* tensor, not ``stiffness_tensor()``'s
        already-rotated one). For a one-off direction different from this
        material's default; see ``stiffness_field_oriented`` for a per-voxel
        orientation field instead of a single direction.

        Parameters
        ----------
        fiber_dir : (3,) unit vector (global frame)
        """
        R = rotation_from_direction(jnp.asarray(fiber_dir, float))
        return rotate_tensor4(R, self._stiffness_tensor_reference())

    def stiffness_field_oriented(self, orientations: jnp.ndarray) -> jnp.ndarray:
        """
        Per-voxel rotated stiffness field, vmapped over a fibre orientation
        field -- the field-valued counterpart of ``stiffness_tensor_rotated``
        (one direction) for a spatially varying orientation (e.g. from
        ``utils.io.reader.read_vtu``'s ``orientations`` output). Like
        ``stiffness_tensor_rotated``, this overrides this material's own
        stored ``fiber_dir`` rather than composing with it.

        The reference-frame tensor is computed once and shared across every
        voxel; only the rotation itself (``rotation_from_direction`` +
        ``rotate_tensor4``, both already written to be vmap-safe) runs
        per-voxel.

        Parameters
        ----------
        orientations : (3, Nv)  unit fibre direction per voxel (global frame)

        Returns
        -------
        C_field : (3, 3, 3, 3, Nv)
        """
        C_ref = self._stiffness_tensor_reference()

        def _one(d):
            R = rotation_from_direction(d)
            return rotate_tensor4(R, C_ref)

        C_vox = jax.vmap(_one, in_axes=1)(orientations)   # (Nv, 3, 3, 3, 3)
        return jnp.moveaxis(C_vox, 0, -1)                  # (3, 3, 3, 3, Nv)

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """
        6x6 Voigt form of ``stiffness_tensor()`` -- this material's own
        ``fiber_dir`` orientation (default Z, i.e. the reference frame).

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
