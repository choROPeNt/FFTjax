"""
TexGen export — isotropic voxel grid.

Derives a single voxelsize from the geometry so that dx = dy = dz,
eliminating the anisotropic grid artefacts that cause spurious PFF
cracking when l₀ < max(dx, dy, dz).

Voxelsize is chosen so that the thinnest dimension (Z, controlled by
yarn thickness Dicke) contains at least MIN_VOXELS_Z voxels.  Adjust
MIN_VOXELS_Z to trade resolution for grid size.

Usage (inside TexGen Python environment):
    python scripts/texgen_export_isotropic.py

Output:
    <file>.vtu
"""

from TexGen.Core import *
import math

# ── Geometry (same as BIIAX_200_no-nesting.py) ───────────────────────────────

n_layer = 2

Breite         = 1.55222   # mm  yarn width
Dicke          = 0.17084   # mm  yarn thickness
Stuetzweite    = 0.0       # extra support point width (0 = disabled)
Musterbreite   = 2.01811   # mm  centre-to-centre yarn spacing
Matrixreiche_Zone = 0.08   # mm  resin-rich interlayer

Bedeckungsgrad = Breite / Musterbreite
Faktor         = 1.0 / Bedeckungsgrad

x_Offset_2     = Musterbreite * 0.5
y_Offset_2     = Musterbreite * 0.5

vertikale_Einrueckung = 0.0
vertOffset_12 = Matrixreiche_Zone + 2.0 * Dicke - vertikale_Einrueckung

# ── Domain dimensions ─────────────────────────────────────────────────────────

Lx = 2.0 * Faktor * Breite
Ly = 2.0 * Faktor * Breite
Lz = vertOffset_12 * n_layer - Matrixreiche_Zone

print(f"Domain : Lx={Lx:.4f}  Ly={Ly:.4f}  Lz={Lz:.4f}  mm")

# ── Isotropic voxelsize ───────────────────────────────────────────────────────
# Drive the voxelsize from the thinnest dimension (Z) so all axes share
# the same spacing.  Increase MIN_VOXELS_Z for finer resolution.

MIN_VOXELS_Z = 20          # at least this many voxels through the thickness
voxelsize    = Lz / MIN_VOXELS_Z

x_res = max(1, int(math.ceil(Lx / voxelsize)))
y_res = max(1, int(math.ceil(Ly / voxelsize)))
z_res = max(1, int(math.ceil(Lz / voxelsize)))

print(f"Voxelsize : {voxelsize:.5f} mm  (isotropic)")
print(f"Grid      : {x_res} × {y_res} × {z_res} = {x_res*y_res*z_res:,} voxels")
print(f"Spacing   : dx={Lx/x_res:.5f}  dy={Ly/y_res:.5f}  dz={Lz/z_res:.5f}  mm")

# ── Build textile ─────────────────────────────────────────────────────────────

Textile = CTextile()
Yarns   = [CYarn() for _ in range(4 * n_layer)]

for i in range(n_layer):
    z0 = i * vertOffset_12

    # weft (X-direction)
    for xn, yn, zn in [
        (x_Offset_2 - 1.0*Faktor*Breite,                         y_Offset_2, 0.5*Dicke + z0),
        (x_Offset_2 - 1.0*Faktor*Breite + Stuetzweite*Breite/2., y_Offset_2, 0.5*Dicke + z0),
        (x_Offset_2 - 0.5*Faktor*Breite,                         y_Offset_2, 1.0*Dicke + z0),
        (x_Offset_2 + 0.0*Faktor*Breite - Stuetzweite*Breite/2., y_Offset_2, 1.5*Dicke + z0),
        (x_Offset_2 + 0.0*Faktor*Breite,                         y_Offset_2, 1.5*Dicke + z0),
        (x_Offset_2 + 0.0*Faktor*Breite + Stuetzweite*Breite/2., y_Offset_2, 1.5*Dicke + z0),
        (x_Offset_2 + 0.5*Faktor*Breite,                         y_Offset_2, 1.0*Dicke + z0),
        (x_Offset_2 + 1.0*Faktor*Breite - Stuetzweite*Breite/2., y_Offset_2, 0.5*Dicke + z0),
        (x_Offset_2 + 1.0*Faktor*Breite,                         y_Offset_2, 0.5*Dicke + z0),
    ]:
        Yarns[2*i].AddNode(CNode(XYZ(xn, yn, zn)))

    # fill (Y-direction)
    for xn, yn, zn in [
        (x_Offset_2, y_Offset_2 - 1.0*Faktor*Breite,                         1.5*Dicke + z0),
        (x_Offset_2, y_Offset_2 - 1.0*Faktor*Breite + Stuetzweite*Breite/2., 1.5*Dicke + z0),
        (x_Offset_2, y_Offset_2 - 0.5*Faktor*Breite,                         1.0*Dicke + z0),
        (x_Offset_2, y_Offset_2 + 0.0*Faktor*Breite - Stuetzweite*Breite/2., 0.5*Dicke + z0),
        (x_Offset_2, y_Offset_2 + 0.0*Faktor*Breite,                         0.5*Dicke + z0),
        (x_Offset_2, y_Offset_2 + 0.0*Faktor*Breite + Stuetzweite*Breite/2., 0.5*Dicke + z0),
        (x_Offset_2, y_Offset_2 + 0.5*Faktor*Breite,                         1.0*Dicke + z0),
        (x_Offset_2, y_Offset_2 + 1.0*Faktor*Breite - Stuetzweite*Breite/2., 1.5*Dicke + z0),
        (x_Offset_2, y_Offset_2 + 1.0*Faktor*Breite,                         1.5*Dicke + z0),
    ]:
        Yarns[2*i+1].AddNode(CNode(XYZ(xn, yn, zn)))

Section = CSectionLenticular(Breite, Dicke * 0.99, 0.0)

for Yarn in Yarns:
    Yarn.AssignInterpolation(CInterpolationCubic())
    Yarn.AssignSection(CYarnSectionConstant(Section))
    Yarn.SetResolution(50)
    Yarn.AddRepeat(XYZ(1.0 * Faktor * Breite, 1.0 * Faktor * Breite, 0))
    Yarn.AddRepeat(XYZ(2.0 * Faktor * Breite, 0, 0))
    Textile.AddYarn(Yarn)

Textile.AssignDomain(CDomainPlanes(
    XYZ(0, 0, 0),
    XYZ(Lx, Ly, Lz),
))

textile_name = f"200_{n_layer:02d}-layer_nn_iso"
AddTextile(textile_name, Textile)

# ── Export ────────────────────────────────────────────────────────────────────

file = f"200_{n_layer:02d}-layer_nn_iso.vtu"

Vox = CRectangularVoxelMesh("CPeriodicBoundaries")
Vox.SaveVoxelMesh(
    GetTextile(textile_name),
    file,
    x_res, y_res, z_res,
    True,   # smooth
    True,   # domain
    0, 0,
    VTU_EXPORT,
)

print(f"Written → {file}")
