import numpy as np


def make_square_composite_rve(phi, r_fiber_um, spacing_um, N_min=32):
    """
    Build the 2-fibre square-cell RVE for given volume fraction and resolution.

    Parameters
    ----------
    phi         : float  target fibre volume fraction  (0 < phi < pi/4 ≈ 0.785)
    r_fiber_um  : float  fibre radius  [µm]
    spacing_um  : float  physical voxel size  [µm/px]
    N_min       : int    minimum grid size per side (default 32)

    Returns
    -------
    phase_np  : ndarray (N, N, 1), int   0 = matrix, 1 = fibre
    N         : int                      voxels per side
    n         : tuple                    (N, N, 1)
    L         : tuple                    (L_side, L_side, 1.0) [µm]
    phi_act   : float                    actual volume fraction on the mesh
    """
    phi_max = np.pi / 4                                # square-packing limit ≈ 0.785
    if phi >= phi_max:
        raise ValueError(f'phi={phi:.4f} exceeds square-packing max {phi_max:.4f}')

    L_side = r_fiber_um * np.sqrt(2 * np.pi / phi)    # cell side [µm]
    N      = max(N_min, int(np.ceil(L_side / spacing_um)))
    n      = (N, N, 1)
    L      = (L_side, L_side, 1.0)

    xs = (np.arange(N) + 0.5) / N * L_side
    X, Y = np.meshgrid(xs, xs, indexing='ij')          # (N, N)

    def _circle(cx, cy):
        dx = X - cx;  dx -= L_side * np.round(dx / L_side)
        dy = Y - cy;  dy -= L_side * np.round(dy / L_side)
        return dx**2 + dy**2 < r_fiber_um**2

    phase_np = (_circle(0.5*L_side, 0.5*L_side) | _circle(0.0, 0.0)).astype(int)
    phase_np = phase_np[:, :, np.newaxis]               # (N, N, 1)
    phi_act  = float(phase_np.mean())

    return phase_np, N, n, L, phi_act
