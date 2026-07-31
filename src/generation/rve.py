import numpy as np


def make_square_composite_rve(phi, r_fiber, spacing, N_min=32, nz=1):
    """
    Build the 2-fibre square-cell RVE for given volume fraction and resolution.

    The in-plane (XY) microstructure is a square-packed fibre arrangement; it is
    extruded uniformly along Z to give a prismatic 3-D domain.

    Unit-agnostic: `r_fiber` and `spacing` may be given in any consistent
    SI-derived length unit (mm preferred, matching this project's other
    mm/MPa/N-based examples) -- the geometry is built from their ratio, so
    `L`/`phi_act` come out in whatever unit was passed in.

    Parameters
    ----------
    phi         : float  target fibre volume fraction  (0 < phi < pi/4 ≈ 0.785)
    r_fiber     : float  fibre radius  [length unit, e.g. mm]
    spacing     : float  physical voxel size  [same length unit, per voxel]
    N_min       : int    minimum grid size per in-plane side (default 32)
    nz          : int    number of voxels through the thickness (default 1)

    Returns
    -------
    phase_np  : ndarray (N, N, nz), int   0 = matrix, 1 = fibre
    N         : int                       voxels per in-plane side
    n         : tuple                     (N, N, nz)
    L         : tuple                     (L_side, L_side, nz * spacing) [same length unit]
    phi_act   : float                     actual volume fraction on the mesh
    """
    if nz < 1:
        raise ValueError(f'nz must be >= 1, got {nz}')
    phi_max = np.pi / 4                                # square-packing limit ≈ 0.785
    if phi >= phi_max:
        raise ValueError(f'phi={phi:.4f} exceeds square-packing max {phi_max:.4f}')

    L_side = r_fiber * np.sqrt(2 * np.pi / phi)       # in-plane cell side [length unit]
    N      = max(N_min, int(np.ceil(L_side / spacing)))
    n      = (N, N, nz)
    L      = (L_side, L_side, nz * spacing)

    xs = (np.arange(N) + 0.5) / N * L_side
    X, Y = np.meshgrid(xs, xs, indexing='ij')          # (N, N)

    def _circle(cx, cy):
        dx = X - cx;  dx -= L_side * np.round(dx / L_side)
        dy = Y - cy;  dy -= L_side * np.round(dy / L_side)
        return dx**2 + dy**2 < r_fiber**2

    phase_2d = (_circle(0.5*L_side, 0.5*L_side) | _circle(0.0, 0.0)).astype(int)
    phase_np = np.repeat(phase_2d[:, :, np.newaxis], nz, axis=2)  # (N, N, nz)
    phi_act  = float(phase_np.mean())

    return phase_np, N, n, L, phi_act
