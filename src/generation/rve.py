"""Composite RVE generators: square-packed and random (Catalanotti 2016) fiber cells."""

import numpy as np


def _remove_isolated_voxels(phase_2d):
    """
    One pass reassigning voxels that differ from all 4 periodic in-plane
    neighbors to the neighbors' majority phase (ties: smallest phase).
    Cleans stray voxels of thin interphase rings. 2-D only: Z is extruded.
    """
    up    = np.roll(phase_2d, -1, axis=0)
    down  = np.roll(phase_2d,  1, axis=0)
    left  = np.roll(phase_2d, -1, axis=1)
    right = np.roll(phase_2d,  1, axis=1)

    isolated = (phase_2d != up) & (phase_2d != down) & (phase_2d != left) & (phase_2d != right)

    cleaned = phase_2d.copy()
    for i, j in np.argwhere(isolated):
        neighbours = (up[i, j], down[i, j], left[i, j], right[i, j])
        vals, counts = np.unique(neighbours, return_counts=True)
        cleaned[i, j] = vals[np.argmax(counts)]
    return cleaned


def make_square_composite_rve(phi, r_fiber, dx, N_min=32, nz=1, clean_isolated=True):
    """
    2-fiber square-packed RVE extruded along Z; units follow r_fiber and dx.

    Parameters
    ----------
    phi            : target fiber volume fraction, < pi/4
    r_fiber        : fiber radius
    dx             : target voxel size; N is rounded up, so the in-plane voxel
                     may be finer (only Z matches dx exactly)
    N_min          : minimum voxels per in-plane side
    nz             : voxels through the thickness
    clean_isolated : remove single-voxel islands

    Returns
    -------
    phase_np : (N, N, nz) int, 0 matrix, 1 fiber
    n, L     : grid shape and length
    phi_act  : actual fiber volume fraction
    """
    if nz < 1:
        raise ValueError(f'nz must be >= 1, got {nz}')
    phi_max = np.pi / 4
    if phi >= phi_max:
        raise ValueError(f'phi={phi:.4f} exceeds square-packing max {phi_max:.4f}')

    L_side = r_fiber * np.sqrt(2 * np.pi / phi)
    N      = max(N_min, int(np.ceil(L_side / dx)))
    n      = (N, N, nz)
    L      = (L_side, L_side, nz * dx)

    xs = (np.arange(N) + 0.5) / N * L_side
    X, Y = np.meshgrid(xs, xs, indexing='ij')

    def _circle(cx, cy):
        dx = X - cx;  dx -= L_side * np.round(dx / L_side)
        dy = Y - cy;  dy -= L_side * np.round(dy / L_side)
        return dx**2 + dy**2 < r_fiber**2

    phase_2d = (_circle(0.5*L_side, 0.5*L_side) | _circle(0.0, 0.0)).astype(int)
    if clean_isolated:
        phase_2d = _remove_isolated_voxels(phase_2d)
    phase_np = np.repeat(phase_2d[:, :, np.newaxis], nz, axis=2)
    phi_act  = float(phase_np.mean())

    return phase_np, n, L, phi_act


def make_random_composite_rve(phi, r_fiber, dx, N_min=32, size_in_r=None, nz=1, K=15,
                               seed=None, interphase_thickness=None, interphase_erosion_fraction=0.0,
                               clean_isolated=True):
    """
    Densely packed random fiber RVE (Catalanotti 2016), any phi up to the
    hexagonal limit 0.9069: hexagonal start, expansion to phi, K random
    non-overlapping moves per fiber.

    Parameters
    ----------
    phi       : target fiber volume fraction
    r_fiber   : fiber radius
    dx        : target voxel size
    N_min     : minimum voxels per side (ignored with size_in_r)
    size_in_r : target side length in fiber radii (e.g. 15)
    nz        : voxels through the thickness
    K         : perturbation rounds (> 10 is fully random)
    seed      : RNG seed (None = not reproducible)
    interphase_thickness : radial thickness of a third phase (2) around each
                  fiber, an exact annulus; None = binary. Packing always uses
                  the nominal r_fiber; a fiber's core wins over neighboring rings
    interphase_erosion_fraction : share of the ring taken from the fiber:
                  0 grows into the matrix (default), 1 eats into the fiber
    clean_isolated : remove single-voxel islands (useful for thin rings)

    Returns
    -------
    phase_np, n, L, phi_act : as make_square_composite_rve; phase 2 is the
                              interphase, phi_act counts fiber only
    centres                 : (Np, 2) fiber centers
    """
    if nz < 1:
        raise ValueError(f'nz must be >= 1, got {nz}')
    if interphase_thickness is not None and interphase_thickness <= 0:
        raise ValueError(f'interphase_thickness must be > 0, got {interphase_thickness}')
    if not 0.0 <= interphase_erosion_fraction <= 1.0:
        raise ValueError(
            f'interphase_erosion_fraction must be in [0, 1], got {interphase_erosion_fraction}'
        )
    phi_max = np.pi * np.sqrt(3.0) / 6.0
    if phi >= phi_max:
        raise ValueError(f'phi={phi:.4f} exceeds hexagonal-packing max {phi_max:.4f}')

    rng = np.random.default_rng(seed)

    delta_x = 2.0 * r_fiber
    delta_y = 2.0 * np.sqrt(3.0) * r_fiber
    f = np.sqrt(phi_max / phi)

    r_core = r_fiber - interphase_erosion_fraction * (interphase_thickness or 0.0)
    r_outer = r_fiber + (1.0 - interphase_erosion_fraction) * (interphase_thickness or 0.0)
    size_floor = size_in_r * r_fiber if size_in_r is not None else N_min * dx
    L_min = max(size_floor, 4.0 * r_outer)
    # >= 3 cells per axis: the single-periodic-image overlap check needs the
    # domain well above the (dilated) fiber diameter.
    N = max(3, int(np.ceil(L_min / (f * delta_y))))
    M = max(3, int(round(N * np.sqrt(3.0))))

    v0 = np.array([0.0, 0.0])
    v1 = np.array([r_fiber, np.sqrt(3.0) * r_fiber])
    centres = np.array([
        v0 + np.array([m * delta_x, n_ * delta_y]) if basis == 0
        else v1 + np.array([m * delta_x, n_ * delta_y])
        for m in range(M) for n_ in range(N) for basis in (0, 1)
    ])

    centres *= f
    Lx, Ly = f * M * delta_x, f * N * delta_y

    Np = centres.shape[0]
    L = np.array([Lx, Ly])
    two_r_sq = (2.0 * r_fiber) ** 2
    rho_cap = 0.5 * min(Lx, Ly)

    for _ in range(K):
        for i in rng.permutation(Np):
            angle = 2.0 * np.pi * rng.uniform()
            direction = np.array([np.cos(angle), np.sin(angle)])

            delta = centres[i] - centres
            delta -= L * np.round(delta / L)
            delta = np.delete(delta, i, axis=0)

            # Largest free move: smallest positive root of |rho d + delta_j|^2 = (2r)^2.
            b = 2.0 * (delta @ direction)
            c = np.sum(delta ** 2, axis=1) - two_r_sq
            disc = b ** 2 - 4.0 * c
            hits = disc >= 0.0
            root = (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0
            root = np.where(hits & (root > 0.0), root, np.inf)
            rho_bar = min(float(np.min(root)), rho_cap)

            rho = rho_bar * rng.uniform()
            centres[i] = (centres[i] + rho * direction) % L

    Nx = max(N_min, int(np.ceil(Lx / dx)))
    Ny = max(N_min, int(np.ceil(Ly / dx)))
    n  = (Nx, Ny, nz)
    L_out = (Lx, Ly, nz * dx)

    xs = (np.arange(Nx) + 0.5) / Nx * Lx
    ys = (np.arange(Ny) + 0.5) / Ny * Ly
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    fibre_mask = np.zeros((Nx, Ny), dtype=bool)
    dilated_mask = np.zeros((Nx, Ny), dtype=bool) if interphase_thickness else None
    for cx, cy in centres:
        ddx = X - cx;  ddx -= Lx * np.round(ddx / Lx)
        ddy = Y - cy;  ddy -= Ly * np.round(ddy / Ly)
        dist_sq = ddx ** 2 + ddy ** 2
        fibre_mask |= dist_sq < r_core ** 2
        if dilated_mask is not None:
            dilated_mask |= dist_sq < r_outer ** 2

    phase_2d = fibre_mask.astype(int)
    if dilated_mask is not None:
        phase_2d[dilated_mask & ~fibre_mask] = 2
    if clean_isolated:
        phase_2d = _remove_isolated_voxels(phase_2d)

    phase_np = np.repeat(phase_2d[:, :, np.newaxis], nz, axis=2)
    phi_act  = float((phase_2d == 1).mean())

    return phase_np, n, L_out, phi_act, centres
