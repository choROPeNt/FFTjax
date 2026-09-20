"""
Phase-field fracture coupled to Drucker-Prager plasticity: a pressure-
sensitive elastoplastic matrix that also cracks.

Why both. A brittle phase-field matrix over-predicts strength and mis-places
the crack in any material that yields first -- an epoxy matrix yields in a
markedly pressure-sensitive way well before it fails. Pure
materialmodels.inelastic.plasticity_drucker_prager.DruckerPrager, on the
other hand, never fails at all: it hardens indefinitely, or plateaus. This
model is the composition, and it is deliberately the THIN one -- the return
mapping below is DruckerPrager's, called through super(), not reimplemented.

The coupling: EFFECTIVE STRESS
------------------------------
The plastic return runs in UNDEGRADED (effective) stress space, exactly as
it does without damage, and ``d`` then scales what the voxel transmits:

    sigma_eff = DruckerPrager.stress(eps, eps_p_prev, alpha_prev)     [unchanged]
    sigma     = g(d) * (s_eff + <p_eff>_+ I)  +  <p_eff>_- I

with g the AT2 degradation (materialmodels.phasefield.degradation.
degradation_at2). That is Amor's split at the stress level: everything
degrades EXCEPT the compressive hydrostatic branch, so a fully cracked voxel
still carries compression instead of interpenetrating. The yield surface
itself never sees ``d``, which is what lets the return mapping -- cone, apex,
tip-rounded, and all of their NaN/q_safe guards -- be reused verbatim rather
than rederived under degradation.

Degrading the yield stress too (sigma_y -> g(d)*sigma_y, so a cracking voxel
also loses strength and plasticity localizes into the crack) is the fuller
coupling and is NOT built. It needs ``d`` threaded through the hardening call
sites, and for a piecewise-affine law it collides with the static
``_c_zero``/``_newton_iters`` conditioning DruckerPrager's affine tip-rounded
path derives at construction -- a traced g(d)*sigma_y invalidates both. See
plasticity_drucker_prager._solve_dlam_hyperbolic.

The crack driving force
-----------------------
    H = < psi_e^+(eps - eps_p)  +  beta_p * W_p  -  psi_c >_+

Two parts and a threshold. ``beta_p`` (default 0) picks the physics:

- ``beta_p = 0`` -- a BRITTLE crack in a plastic solid. Plasticity influences
  fracture only by capping the elastic energy the material can store, which
  blunts the crack. Note psi_e^+ is evaluated on the ELASTIC strain
  eps - eps_p, not the total strain: plastic strain stores no energy, and
  using the total strain here would let plastic flow drive cracking through
  the back door.
- ``beta_p > 0`` -- DUCTILE fracture: accumulated plastic work also drives
  the crack, so a voxel that has yielded heavily cracks sooner.

W_p is accumulated as sigma_eff : d(eps_p), which is exact for associated AND
non-associated flow -- the closed form int sigma_y dalpha is not, since
sigma:deps_p = dlam*(sigma_y + 3*(a_g - a_f)*p) and the two agree only at
a_g = a_f. It is floored at zero per increment; see stress() for the one
(deeply confined, strongly non-associated) regime where that matters.

``psi_c`` -- the damage-initiation threshold, why it exists
----------------------------------------------------------
AT2 has NO elastic threshold. Solve its damage equation on a homogeneous
field, where the Laplacian drops out:

    (Gc/l0 + 2H) d = 2H      =>      d = 2H / (Gc/l0 + 2H)

which is strictly positive for any H > 0. Damage therefore nucleates at
arbitrarily small strain and the stress-strain curve bends off the elastic
line from the ORIGIN -- so a plastic material never reaches its yield stress
before softening starts, which is wrong for any polymer that visibly yields
and necks first.

Raising Gc does not fix it, it only makes the early damage smaller. Measured
on the repo's PP card (E=1300, sigma_y0=36, a_f=0.13 -> sigma_t=31.86 MPa,
so H at yield = sigma_t^2/2E = 0.390 MPa) at l0 = 0.5:

    d at yield   0.20    0.10    0.05    0.01    0.001
    needs Gc     1.6     3.5     7.4     38.7    390.0   N/mm

against a real PP GIc of O(1-10) N/mm -- and at Gc ~ 39 nothing fractures in
any reachable strain range. So the threshold is the fix, not a tuning knob.

Set it as a STRESS, ``sigma_crit`` (MPa), and the model converts:

    psi_c = sigma_crit^2 / (2E)

which is EXACT for uniaxial tension, not an approximation: there every
principal strain contributes tensile energy, so the Amor psi_pos equals the
full elastic energy sigma^2/(2E) identically (verified term by term). For a
general multiaxial state it is the usual energy-equivalent reading of a
uniaxial calibration.

``sigma_crit = sigma_y0/(1 + a_f)`` -- the model's own uniaxial tensile yield
stress -- makes damage start exactly at yield, which is the elastic -> yield
-> plastic flow -> fracture sequence a ductile polymer actually shows. Larger
values delay cracking further into the plastic regime.

The threshold applies to the TOTAL driving force, plastic work included, so a
voxel that has yielded can be pushed over it by W_p alone once beta_p > 0.
It is the standard device for giving AT2 an initiation barrier (Miehe et al.
2010 and successors) rather than something derived here; note it does make
the model non-variational in the strict sense, since the threshold is not
itself the derivative of the AT2 functional. The principled alternative is
AT1 (linear w(d)), which has an intrinsic threshold by construction -- not
implemented in this package, and a larger change than this parameter.

Default is no threshold (``sigma_crit=None``), which is bit-for-bit the
behaviour before this parameter existed.

Three behaviours worth knowing before reading a result
------------------------------------------------------
- The tangent is NON-SYMMETRIC wherever a voxel is both damaged and plastic,
  and increasingly so with d. Measured on associated flow (a_g = a_f), where
  DruckerPrager's own tangent is exactly symmetric, the relative major
  asymmetry |C_ijkl - C_klij|/|C| runs

      d     = 0      1e-6     0.1      0.5      0.9     0.999
      asym  = 3e-16  5e-07    5e-02    2e-01    3e-01   3e-01

  while a purely ELASTIC state stays symmetric to 1e-16 at any d. So this is
  not degradation alone and not plasticity alone: the Amor stress-level
  operator g*P_+ + P_- commutes with the isotropic elastic tangent but not
  with the elastoplastic one's s⊗s structure, so composing them loses the
  symmetry. Non-associated flow (a_g < a_f) adds DruckerPrager's own
  asymmetry on top.

  This matters because problems.mechanics.solve_displacement_based_nonlinear
  applies CG to this operator unsymmetrized, which is formally outside CG's
  assumptions -- the same caveat DruckerPrager.stress_and_tangent documents
  for a_g != a_f, but here it applies to EVERY coupled material and grows
  with the crack. Do NOT symmetrize the tangent to "fix" this: that would
  break the quadratic Newton convergence it exists to provide.

  WHAT IS KNOWN about this. The asymmetry does cost the inner CG accuracy --
  on the 40^3 two-fibre cell it routinely stops at its 500-iteration cap at
  relative residual 1e-4..1e-6 against toler_lin = 1e-7, from the first
  damaged increment on -- but that has NOT been seen to stop Newton: those
  inexact steps still converge it in 3-8 iterations over that cell's whole
  load path to d_max = 0.96.

  The failure that used to appear past the softening branch -- Newton
  stalling once the transmitted stress was three to four orders below peak,
  and returning a small NEGATIVE macroscopic stress under applied tension --
  was the Newton INITIAL GUESS, not this model. problems.fracture started
  every increment's first staggered sweep from the uniform eps_bar, discarding
  the previous increment's strain fluctuation and its lateral contraction on
  every stress-controlled direction. Deep in plastic flow that start is far
  enough off that the first Newton step drives the plastic return into the
  wrong branch almost everywhere, and Newton then cycles instead of
  converging. Passing the previous increment's strain as ``eps_init`` (which
  solve_fracture_incremental now does) removes it. On check 8's seeded 16^3
  cube, pulled on to eps_11 = 0.05 in 25 equal increments: every increment
  converges, Newton in 3-4 iterations, sigma_11 falling smoothly and staying
  positive down to 3.6e-4 of peak. From the uniform guess the same increments
  are not even reproducible, because a cycling Newton amplifies round-off: in
  two identical runs eps_11 = 0.010 returned sigma_11 = 0.050 and 0.058, and
  eps_11 = 0.012 returned +0.005 (converged) in one and -0.73 (Newton failed)
  in the other. The 40^3 two-fibre cell measurement is in
  problems.fracture.solve_fracture's ``eps_init`` note.

  Four other explanations had been tested earlier and ruled out, each on an
  isolated reproduction, which is consistent with that: severance making the
  relative Newton residual unreachable (a fully severing plane converges in
  3 iterations at rel 2e-9 for d_plane up to 0.999999); the ~1e6 stiffness
  contrast (a fixed localized crack at d = 0.9999 converges in 8); the C0
  preconditioner choice (identical to every printed digit); and plastic
  strain turning the crack into a compressive strut (tr(eps - eps_p) stays
  positive and growing under monotone tension at d = 0.9999). None of those
  reproductions started Newton cold from a plastic state.

  problems.fracture still folds a mechanical failure into the increment's
  convergence flag, so a load stepper in "automatic" mode cuts back on it
  rather than accepting a non-equilibrium state and carrying its plastic
  state forward. A caller that drives solve_fracture directly, increment by
  increment, has to thread ``eps_init`` itself.
- The tangent also JUMPS across p_eff = 0 (g*K against K). That one is not
  new here: materialmodels.phasefield.isotropic.PhaseFieldIsotropic has
  exactly the same jump from the same Amor structure. It is what a
  tension/compression split means, not a defect.
- A fully cracked voxel keeps its EFFECTIVE stress on the yield surface, so
  its plastic strain goes on accumulating while it transmits almost nothing.
  That is inherent to effective-stress ductile phase-field, and benign here
  because H has long saturated there -- but it does mean eps_p and alpha
  inside a crack are not physically meaningful, and a homogenized plastic
  strain that includes cracked voxels will read high.

State and dispatch
------------------
State is the 3-tuple ``(eps_p, alpha, W_p)``, owned by the CALLER across
increments exactly as DruckerPrager's 2-tuple is, and ``d`` is likewise never
stored here. problems.mechanics.solve_displacement_based_nonlinear is
state-agnostic ("this driver only carries it from iteration to iteration"),
so the extra slot needs no change to it.

``driving_force_field`` is the duck-type marker for "stateful phase-field
material", the way ``psi_split`` marks a PhaseFieldIsotropic-style one and
``stress_and_tangent_field`` marks a stateful one --
materialmodels.assembly.assemble_inelastic_pff_local_update dispatches on it,
and the other two combinators raise on it rather than silently mishandling
it.

Note the methods below deliberately SHADOW DruckerPrager's with a different
signature (``d`` added, 3-tuple state). A 3-argument call therefore fails
with a TypeError instead of quietly running the undegraded model, which is
the failure mode worth protecting against -- assemble_local_update, which
duck-types on ``stress_and_tangent_field``, would otherwise return
completely undamaged stresses for this material.
"""

import sys

import jax
import jax.numpy as jnp

from materialmodels.inelastic.plasticity_drucker_prager import DruckerPrager
from materialmodels.phasefield.degradation import degradation_at2
from materialmodels.phasefield.splits import amor_split


class PhaseFieldDruckerPrager(DruckerPrager):
    """
    Drucker-Prager plasticity with AT2 phase-field damage, coupled through
    the effective stress (see the module docstring).

    Parameters
    ----------
    E, nu     : float  isotropic elastic constants
    Gc        : float  critical energy release rate -- REQUIRED here, unlike
                on DruckerPrager where it is optional bookkeeping. Same
                reasoning as PhaseFieldIsotropic: a phase-field material with
                no Gc is a config mistake worth catching at construction
                rather than deep inside a fracture solve.
    sigma_y0, H, hardening, a_f, a_g, a_tip : as DruckerPrager -- a_f is
                still required, and any hardening law is accepted
    beta_p    : float  weight on accumulated plastic work in the crack
                driving force, >= 0. 0 (default) is a brittle crack in a
                plastic solid; > 0 is ductile fracture. Dimensionless.
    sigma_crit : float | None  damage-initiation stress (MPa) -- below this,
                nothing damages. None (default) means no threshold, which is
                AT2's own behaviour: damage from arbitrarily small strain. Set
                it to ``sigma_y0/(1 + a_f)`` to start damage exactly at yield.
                See the module docstring for why this exists and why raising
                Gc is not an alternative.
    k_res     : float  AT2 residual stiffness
    name      : str    optional label
    """

    def __init__(self, E: float, nu: float, Gc: float,
                 sigma_y0: float | None = None, H: float | None = None,
                 a_f: float | None = None, a_g: float | None = None,
                 a_tip: float = 0.0,
                 hardening=None, beta_p: float = 0.0,
                 sigma_crit: float | None = None,
                 name: str = "", k_res: float = 1e-6):
        super().__init__(E, nu, sigma_y0=sigma_y0, H=H, a_f=a_f, a_g=a_g,
                         a_tip=a_tip, hardening=hardening, name=name,
                         k_res=k_res, Gc=Gc)
        self.beta_p = float(beta_p)
        if self.beta_p < 0.0:
            raise ValueError(
                f"PhaseFieldDruckerPrager {self.name!r}: beta_p={self.beta_p} < 0 "
                "-- a negative weight would make plastic work HEAL the crack"
            )
        # Stored as the stress the user calibrates, and as the energy density
        # the driving force actually subtracts. Exact for uniaxial tension --
        # see the module docstring.
        self.sigma_crit = None if sigma_crit is None else float(sigma_crit)
        if self.sigma_crit is None:
            self.psi_c = 0.0
        else:
            if self.sigma_crit < 0.0:
                raise ValueError(
                    f"PhaseFieldDruckerPrager {self.name!r}: "
                    f"sigma_crit={self.sigma_crit} < 0 -- a negative initiation "
                    "stress has no meaning; use None for no threshold"
                )
            self.psi_c = self.sigma_crit ** 2 / (2.0 * self.E)
            # A threshold above the material's own tensile yield stress means
            # the voxel must yield BEFORE it can damage -- legitimate (that is
            # ductile fracture), but worth saying out loud, because combined
            # with beta_p = 0 the elastic energy alone may then never reach it
            # and nothing will ever crack.
            sig_t = self.sigma_y0 / (1.0 + self.a_f)
            if self.sigma_crit > sig_t and self.beta_p == 0.0:
                print(
                    f"WARNING: PhaseFieldDruckerPrager {self.name!r}: "
                    f"sigma_crit={self.sigma_crit:.4g} exceeds the uniaxial tensile "
                    f"yield stress sigma_y0/(1+a_f)={sig_t:.4g}, with beta_p=0.\n"
                    f"         Past yield the elastic energy grows only as fast as "
                    f"hardening allows, so the threshold may never be\n"
                    f"         reached and the material may never crack. Give "
                    f"beta_p > 0 so plastic work can drive it, or lower sigma_crit.",
                    file=sys.stderr)

    # ── the crack driving force ─────────────────────────────────────────
    def psi_split(self, eps_e: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Undegraded Amor split of the ELASTIC strain, one voxel: (3,3) ->
        (psi_pos, psi_neg). Note the argument is eps - eps_p, not the total
        strain -- this class's callers do that subtraction; see
        driving_force. Shares the single-voxel formula in
        materialmodels.phasefield.splits with PhaseFieldIsotropic, so the two
        cannot drift.
        """
        return amor_split(eps_e, self.lam, self.mu)

    def driving_force(self, eps, eps_p, W_p):
        """
        One voxel: (drive, psi_pos) from the CURRENT state.

            psi_pos = amor_split(sym(eps) - eps_p)[0]
            drive   = max(psi_pos + beta_p * W_p - psi_c, 0)

        Both are returned because they answer different questions: ``drive``
        is what the damage sub-solve consumes, ``psi_pos`` is the elastic
        tensile energy a caller writes out as ``strain_energy_pos``. Returning
        only the sum would silently redefine that output field once beta_p or
        psi_c is nonzero.

        The max(., 0) is what gives AT2 an elastic threshold it otherwise
        lacks entirely -- see the module docstring. At psi_c = 0 (the default)
        it is inactive, since psi_pos and W_p are both non-negative.
        """
        eps_e = 0.5 * (eps + eps.T) - eps_p
        psi_pos, _ = self.psi_split(eps_e)
        drive = jnp.maximum(psi_pos + self.beta_p * W_p - self.psi_c, 0.0)
        return drive, psi_pos

    def driving_force_field(self, eps_field, eps_p_field, W_p_field):
        """
        Field-level ``driving_force``, vmapped over the trailing voxel axis.

        Its PRESENCE is the duck-type marker the assembly layer dispatches on
        (see the module docstring), so renaming it silently changes which
        combinator path this material takes.

        Parameters
        ----------
        eps_field   : (3, 3, Nv)
        eps_p_field : (3, 3, Nv)
        W_p_field   : (Nv,)

        Returns
        -------
        drive_field, psi_pos_field : (Nv,), (Nv,)
        """
        return jax.vmap(self.driving_force)(
            eps_field.transpose(2, 0, 1), eps_p_field.transpose(2, 0, 1), W_p_field
        )

    # ── degraded stress / tangent ───────────────────────────────────────
    def stress(self, eps, d, eps_p_prev, alpha_prev, W_p_prev):
        """
        One-voxel degraded return mapping.

        ``d`` sits SECOND, right after the strain, so a call written for
        DruckerPrager's (eps, eps_p, alpha) signature cannot line up by
        accident -- it raises for a missing argument instead.

        Returns
        -------
        sigma                 : (3,3)   degraded (transmitted) stress
        (eps_p, alpha, W_p)   : updated state
        """
        eps = 0.5 * (eps + eps.T)
        I3 = jnp.eye(3)

        # The plastic return, undegraded and untouched -- every branch and
        # guard in DruckerPrager applies unchanged because d appears nowhere
        # in it. It symmetrizes eps again internally; harmless, and it keeps
        # this call correct if ever made directly.
        sigma_eff, (eps_p, alpha) = super().stress(eps, eps_p_prev, alpha_prev)

        p_eff = jnp.trace(sigma_eff) / 3.0
        s_eff = sigma_eff - p_eff * I3
        g = degradation_at2(d, k=self.k_res)

        # Amor at the stress level: the deviator and any TENSILE pressure
        # degrade, the compressive pressure does not. Written as
        # g*<p>_+ + <p>_- rather than g*p + (1-g)*<p>_- so the two branches
        # are visibly exclusive.
        sigma = g * (s_eff + jnp.maximum(p_eff, 0.0) * I3) \
            + jnp.minimum(p_eff, 0.0) * I3

        # Accumulated plastic work, from the EFFECTIVE stress -- the standard
        # choice in effective-stress ductile phase-field, and the one
        # consistent with the return above.
        #
        # Floored at 0. sigma_eff : deps_p = dlam*(sigma_y + 3*(a_g - a_f)*p)
        # is non-negative for associated flow, but a NON-associated material
        # (a_g < a_f) confined past p = -sigma_y/(3*(a_f - a_g)) can make it
        # negative -- there the non-associated flow rule is no longer
        # dissipative and the "work" is not a legitimate crack driver, so it
        # contributes nothing rather than un-driving the crack. For the
        # config in configs/simulation/, a_f - a_g = 0.08 and sigma_y ~ 56,
        # that bound is p ~ -233 MPa.
        dW_p = jnp.maximum(jnp.sum(sigma_eff * (eps_p - eps_p_prev)), 0.0)
        W_p = W_p_prev + dW_p

        return sigma, (eps_p, alpha, W_p)

    def stress_and_tangent(self, eps, d, eps_p_prev, alpha_prev, W_p_prev):
        """
        One-voxel degraded stress, consistent tangent, and updated state --
        tangent by jax.jacfwd on stress(), same one-pass has_aux pattern as
        both plasticity models and PhaseFieldIsotropic.

        ``d`` is held FIXED here: it is the staggered scheme's frozen damage
        for this mechanical sub-solve, so dsigma/d(eps) at fixed d is exactly
        the tangent the Newton solve needs. The tangent inherits
        DruckerPrager's asymmetry whenever a_g != a_f (see
        DruckerPrager.stress_and_tangent), and gains the p_eff = 0 jump the
        module docstring describes.

        Returns
        -------
        sigma : (3,3)
        C_tan : (3,3,3,3)
        (eps_p, alpha, W_p) : updated state
        """
        def _fn(e):
            sigma, new_state = self.stress(e, d, eps_p_prev, alpha_prev, W_p_prev)
            return sigma, (sigma, new_state)

        C_tan, (sigma, new_state) = jax.jacfwd(_fn, has_aux=True)(eps)
        return sigma, C_tan, new_state

    def stress_and_tangent_field(self, eps_field, d_field, eps_p_field,
                                 alpha_field, W_p_field):
        """
        Per-voxel degraded stress/tangent/state field, vmapped over
        stress_and_tangent.

        Five arguments, where DruckerPrager's takes three -- deliberately not
        call-compatible, so materialmodels.assembly.assemble_local_update
        (which duck-types on this method's NAME) cannot silently drive this
        material undegraded. That combinator also rejects this class
        explicitly, with a message; this signature is the backstop.

        Parameters
        ----------
        eps_field   : (3, 3, Nv)
        d_field     : (Nv,)
        eps_p_field : (3, 3, Nv)
        alpha_field : (Nv,)
        W_p_field   : (Nv,)

        Returns
        -------
        sigma_field : (3, 3, Nv)
        C_field     : (3, 3, 3, 3, Nv)
        (eps_p_field, alpha_field, W_p_field) : updated state
        """
        sigma_vox, C_vox, (eps_p_vox, alpha_new, W_p_new) = jax.vmap(
            self.stress_and_tangent
        )(eps_field.transpose(2, 0, 1), d_field,
          eps_p_field.transpose(2, 0, 1), alpha_field, W_p_field)

        return (sigma_vox.transpose(1, 2, 0),
                C_vox.transpose(1, 2, 3, 4, 0),
                (eps_p_vox.transpose(1, 2, 0), alpha_new, W_p_new))

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        flow = "associated" if self.a_g == self.a_f else "non-associated"
        tip = (f", a_tip={self.a_tip:.3g}" if self.a_tip > 0.0 else " (sharp tip)")
        mode = "brittle" if self.beta_p == 0.0 else f"ductile, beta_p={self.beta_p:.3g}"
        thr = ("" if self.sigma_crit is None
               else f", sigma_crit={self.sigma_crit:.4g} (psi_c={self.psi_c:.4g})")
        return (f"PhaseFieldDruckerPrager{tag}: E={self.E:.3g}, nu={self.nu:.3g}, "
                f"Gc={self.Gc:.3g}, hardening={self.hardening}, "
                f"a_f={self.a_f:.3g}, a_g={self.a_g:.3g} ({flow}){tip}{thr}, "
                f"k_res={self.k_res:.3g} [{mode}]")
