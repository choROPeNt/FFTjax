# ⚙️ Mechanical Solvers

FFTjax provides Newton-CG solvers for the mechanical equilibrium problem (`solvers/mechanical/`),
covering both linear elastic and nonlinear (J2 plastic) materials. The linear variants differ in
which field is the CG unknown; the plastic solver extends the displacement-based formulation with
a genuine outer Newton loop.

## Linear elastic formulations

**Strain-based (`strain_nw_cg.py`, Vondřejc et al. 2014 / Lucarini & Segurado 2018).**
The unknown is the periodic strain fluctuation $\Delta\varepsilon$. Equilibrium is enforced through a fixed reference-medium Green's operator $\hat{\Gamma}_0(\boldsymbol{\xi})$ (isotropic, "standard" or Willot "rotated" discretisation):

$$
A(\Delta\varepsilon) = \mathcal{F}^{-1}\!\big[\hat{\Gamma}_0 : \mathcal{F}[\mathbb{C}:\Delta\varepsilon]\big], \qquad
b = -\mathcal{F}^{-1}\!\big[\hat{\Gamma}_0 : \mathcal{F}[\mathbb{C}:\varepsilon_0 - \sigma_{\text{goal}}]\big]
$$

$\hat{\Gamma}_0$ only ever appears in *even* powers of $\boldsymbol{\xi}$ (e.g. $\xi_i\xi_j$), which is what makes the CG operator provably symmetric positive-definite for arbitrarily heterogeneous $\mathbb{C}(\mathbf{x})$, and is why it needs no extra preconditioning beyond the reference medium itself.

**Displacement-based (`displacement_nw_cg.py`).**
The unknown is instead the periodic displacement fluctuation $\hat{u}$ itself, with the strain built directly from its symmetric gradient in Fourier space, $\hat{\varepsilon} = \tfrac{1}{2}(i\boldsymbol{\xi}\otimes\hat{u} + \text{sym.})$, and the true (possibly heterogeneous) tangent $\mathbb{C}(\mathbf{x})$ applied with **no reference-medium approximation**. Because $\boldsymbol{\xi}$ now appears to an *odd* power, the Nyquist frequency component must be zeroed on even grid dimensions to preserve the Hermitian symmetry a real-input FFT requires — an FFT-Galerkin-specific correction not needed by the Green's-operator formulation. Convergence for realistic (high-contrast) composites additionally needs a frequency-domain preconditioner $(\boldsymbol{\xi}\cdot\mathbb{C}_0\cdot\boldsymbol{\xi})^{-1}$ built from a reference stiffness $\mathbb{C}_0$.

**Mixed strain/stress boundary conditions.**
Both formulations can prescribe macroscopic strain on some tensor components and macroscopic stress on others (e.g. a free-lateral-surface uniaxial-stress test), selected via a `(3,3)` `control` mask. The two schemes solve this differently:

- The *displacement-based* solver embeds the stress-controlled average-strain correction directly as an extra unknown in the zero-frequency mode of the strain field, coupled to the fluctuation field through the same symmetric CG operator — valid for arbitrary (including heterogeneous) materials.
- The *strain-based* solver can instead reclaim the Green's operator's own (otherwise always-zero) DC block for this purpose (`dstrain_nw_cg_mixed`), which is cheaper but only symmetric — and therefore only a valid one-shot CG solve — for **homogeneous** materials; for heterogeneous materials the mixed-BC CG operator loses symmetry and the displacement-based solver must be used instead.

---

## Nonlinear extension: J2 plasticity

`displacement_nw_plastic.py` generalizes the displacement-based solver from a one-shot linear CG
solve into a genuine outer Newton loop for small-strain **J2 (von Mises) plasticity with linear
isotropic hardening**.

**Local constitutive update (radial return, `mat_models/plastic.py`).**
Because the yield function is linear in the plastic multiplier for linear isotropic hardening, the
local return mapping has a closed form — no inner Newton loop per voxel is needed (de Souza Neto,
Peric & Owen, *Computational Methods for Plasticity*, Box 8.1):

$$
\sigma_{\text{trial}} = \lambda\,\mathrm{tr}(\varepsilon - \varepsilon_p^{n})\,\mathbf{I} + 2\mu(\varepsilon - \varepsilon_p^{n}),
\qquad
q_{\text{trial}} = \sqrt{\tfrac{3}{2}}\,\lVert s_{\text{trial}}\rVert,
\qquad
f_{\text{trial}} = q_{\text{trial}} - (\sigma_{y0} + H\alpha^{n})
$$

If $f_{\text{trial}} \le 0$ the step is elastic. Otherwise the plastic multiplier
$\Delta\gamma = f_{\text{trial}} / (3\mu + H)$ radially returns the deviatoric trial stress onto
the yield surface, updating the plastic strain $\varepsilon_p$ and accumulated equivalent plastic
strain $\alpha$. The return map also produces the **algorithmic (consistent) tangent**
$\mathbb{C}_{\text{algo}} = \partial\sigma/\partial\varepsilon$ needed by the outer Newton solve —
not simply $\mathbb{C}:\varepsilon$. This tangent is, by construction, discontinuous across the
yield surface (it jumps from the elastic $\mathbb{C}_e$ to the plastic branch exactly at first
yield); that discontinuity is expected, not a defect.

**Outer Newton-CG loop.**
Structurally this reuses the same FFT/preconditioner scaffolding as the linear displacement-based
solver (same DC-embedding trick for stress-controlled macroscopic directions, same
reference-medium preconditioner $(\boldsymbol{\xi}\cdot\mathbb{C}_0\cdot\boldsymbol{\xi})^{-1}$),
but at each Newton iteration $k$:

- the **residual** is driven by the *true* nonlinear stress field from `j2_return_map_field`
  evaluated at the current strain estimate $\varepsilon_k$, not a linearized approximation;
- the **inner CG solve** linearizes against the *algorithmic tangent* $\mathbb{C}_{\text{algo}}(\varepsilon_k)$ at that same strain estimate.

This is deliberately not the same as calling the linear elastic solver repeatedly with an updated
`C_field`: that would silently re-derive the baseline strain from `eps_bar` on every call instead
of carrying the accumulated strain field forward between Newton iterations, breaking the path
dependence that plasticity relies on. Here $\varepsilon_k$ and the stress-controlled macroscopic
strain correction are both carried across Newton iterations and used as the warm start for the
next load increment, while the plastic strain $\varepsilon_p$ and hardening variable $\alpha$ are
frozen during an increment's Newton iterations and only committed once Newton converges — the
increment-to-increment path dependence plasticity requires.

The plastic solver is kept as a separate module rather than folded into `displacement_nw_cg.py`,
matching this project's practice of keeping newer, less battle-tested solver variants alongside
the verified linear ones rather than generalizing them in place.
