import numpy as np


def make_square_composite_rve(phi, r_fiber, dx, N_min=32, nz=1):
    """
    Build the 2-fibre square-cell RVE for given volume fraction and resolution.

    The in-plane (XY) microstructure is a square-packed fibre arrangement; it is
    extruded uniformly along Z to give a prismatic 3-D domain.

    Unit-agnostic: `r_fiber` and `dx` may be given in any consistent
    SI-derived length unit (mm preferred, matching this project's other
    mm/MPa/N-based examples) -- the geometry is built from their ratio, so
    `L`/`phi_act` come out in whatever unit was passed in.

    Parameters
    ----------
    phi         : float  target fibre volume fraction  (0 < phi < pi/4 ≈ 0.785)
    r_fiber     : float  fibre radius  [length unit, e.g. mm]
    dx          : float  target voxel size [same length unit] -- a target, not a
                  guarantee: N is rounded up (see N_min below), so the realized
                  in-plane voxel size L_side/N can be finer than dx; only the Z
                  voxel size (nz*dx / nz) comes out exactly equal to dx.
    N_min       : int    minimum grid size per in-plane side (default 32)
    nz          : int    number of voxels through the thickness (default 1)

    Returns
    -------
    phase_np  : ndarray (N, N, nz), int   0 = matrix, 1 = fibre
    n         : tuple                     (N, N, nz)
    L         : tuple                     (L_x, L_y, L_z) [same length unit]
    phi_act   : float                     actual volume fraction on the mesh
    """
    if nz < 1:
        raise ValueError(f'nz must be >= 1, got {nz}')
    phi_max = np.pi / 4                                # square-packing limit ≈ 0.785
    if phi >= phi_max:
        raise ValueError(f'phi={phi:.4f} exceeds square-packing max {phi_max:.4f}')

    L_side = r_fiber * np.sqrt(2 * np.pi / phi)       # in-plane cell side [length unit]
    N      = max(N_min, int(np.ceil(L_side / dx)))
    n      = (N, N, nz)
    L      = (L_side, L_side, nz * dx)

    xs = (np.arange(N) + 0.5) / N * L_side
    X, Y = np.meshgrid(xs, xs, indexing='ij')          # (N, N)

    def _circle(cx, cy):
        dx = X - cx;  dx -= L_side * np.round(dx / L_side)
        dy = Y - cy;  dy -= L_side * np.round(dy / L_side)
        return dx**2 + dy**2 < r_fiber**2

    phase_2d = (_circle(0.5*L_side, 0.5*L_side) | _circle(0.0, 0.0)).astype(int)
    phase_np = np.repeat(phase_2d[:, :, np.newaxis], nz, axis=2)  # (N, N, nz)
    phi_act  = float(phase_np.mean())

    return phase_np, n, L, phi_act


def make_random_composite_rve(phi, r_fiber, dx, N_min=32, nz=1, K=15, seed=None):
    """
    Densely-packed random-fibre RVE. Based on Catalanotti (2016),
    doi.org/10.1016/j.compstruct.2015.11.039 -- reaches any phi up to the
    hexagonal limit (~0.9069).

    Unit-agnostic like make_square_composite_rve.

    Parameters
    ----------
    phi     : target fibre volume fraction (0 < phi < pi*sqrt(3)/6 ≈ 0.9069)
    r_fiber : fibre radius  [length unit, e.g. mm]
    dx      : target voxel size [same length unit] -- a target, not a guarantee
    N_min   : minimum grid size per in-plane side (default 32)
    nz      : number of voxels through the thickness (default 1)
    K       : perturbation iterations (default 15; K>10 -> fully random)
    seed    : RNG seed (None -> fresh, non-reproducible sequence)

    Returns
    -------
    phase_np, n, L, phi_act, centres -- see make_square_composite_rve for the
    first four; centres is (Np, 2), the final fibre centre coordinates.
    """
    if nz < 1:
        raise ValueError(f'nz must be >= 1, got {nz}')
    phi_max = np.pi * np.sqrt(3.0) / 6.0   # hexagonal close-packing limit ≈ 0.9069
    if phi >= phi_max:
        raise ValueError(f'phi={phi:.4f} exceeds hexagonal-packing max {phi_max:.4f}')

    rng = np.random.default_rng(seed)

    # ── step 1: compact RVE (densest hexagonal packing) ─────────────────────
    delta_x = 2.0 * r_fiber
    delta_y = 2.0 * np.sqrt(3.0) * r_fiber
    f = np.sqrt(phi_max / phi)   # uniform expansion factor to reach target phi (step 2)

    L_min = max(N_min * dx, 4.0 * r_fiber)
    # >=3 cells per axis (>=18 fibres): keeps the domain comfortably larger
    # than the fibre diameter for the single-periodic-image check below --
    # see docstring.
    N = max(3, int(np.ceil(L_min / (f * delta_y))))
    M = max(3, int(round(N * np.sqrt(3.0))))   # ~square compact rectangle (M*delta_x ~ N*delta_y)

    v0 = np.array([0.0, 0.0])
    v1 = np.array([r_fiber, np.sqrt(3.0) * r_fiber])
    centres = np.array([
        v0 + np.array([m * delta_x, n_ * delta_y]) if basis == 0
        else v1 + np.array([m * delta_x, n_ * delta_y])
        for m in range(M) for n_ in range(N) for basis in (0, 1)
    ])   # (Np, 2), Np = 2*M*N

    # ── step 2: expand to the target volume fraction ────────────────────────
    centres *= f
    Lx, Ly = f * M * delta_x, f * N * delta_y

    # ── step 3: perturbation ─────────────────────────────────────────────────
    Np = centres.shape[0]
    L = np.array([Lx, Ly])
    two_r_sq = (2.0 * r_fiber) ** 2
    rho_cap = 0.5 * min(Lx, Ly)   # sane upper bound on a single move

    for _ in range(K):
        for i in rng.permutation(Np):
            angle = 2.0 * np.pi * rng.uniform()
            direction = np.array([np.cos(angle), np.sin(angle)])

            delta = centres[i] - centres            # (Np, 2), periodic offset to every fibre
            delta -= L * np.round(delta / L)
            delta = np.delete(delta, i, axis=0)      # drop self

            # smallest positive root of |rho*direction + delta_j|^2 = (2r)^2
            b = 2.0 * (delta @ direction)
            c = np.sum(delta ** 2, axis=1) - two_r_sq
            disc = b ** 2 - 4.0 * c
            hits = disc >= 0.0
            root = (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0
            root = np.where(hits & (root > 0.0), root, np.inf)
            rho_bar = min(float(np.min(root)), rho_cap)

            rho = rho_bar * rng.uniform()
            centres[i] = (centres[i] + rho * direction) % L

    # ── voxelise ─────────────────────────────────────────────────────────────
    Nx = max(N_min, int(np.ceil(Lx / dx)))
    Ny = max(N_min, int(np.ceil(Ly / dx)))
    n  = (Nx, Ny, nz)
    L_out = (Lx, Ly, nz * dx)

    xs = (np.arange(Nx) + 0.5) / Nx * Lx
    ys = (np.arange(Ny) + 0.5) / Ny * Ly
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    phase_2d = np.zeros((Nx, Ny), dtype=bool)
    for cx, cy in centres:
        ddx = X - cx;  ddx -= Lx * np.round(ddx / Lx)
        ddy = Y - cy;  ddy -= Ly * np.round(ddy / Ly)
        phase_2d |= (ddx ** 2 + ddy ** 2 < r_fiber ** 2)

    phase_np = np.repeat(phase_2d.astype(int)[:, :, np.newaxis], nz, axis=2)
    phi_act  = float(phase_np.mean())

    return phase_np, n, L_out, phi_act, centres
