"""Build a ConstitutiveModel from a YAML config entry, selected by its ``model`` key."""

from materialmodels.base import ConstitutiveModel
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from materialmodels.inelastic.hardening import build_hardening
from materialmodels.inelastic.plasticity_drucker_prager import DruckerPrager
from materialmodels.inelastic.plasticity_j2 import J2Plasticity
from materialmodels.phasefield.isotropic import PhaseFieldIsotropic

_MODELS = {
    "isotropic_elastic":    LinearElasticIsotropic,
    "transverse_isotropic": TransverseIsotropic,
    "phasefield_isotropic": PhaseFieldIsotropic,
    "j2_plasticity":        J2Plasticity,
    "drucker_prager":       DruckerPrager,
}


def build_material(cfg: dict, orientations=None) -> ConstitutiveModel:
    """
    Material from one ``materials:`` entry, e.g.
    ``{"model": "isotropic_elastic", "E": 3.0e3, "nu": 0.35, "name": "epoxy"}``.

    Every key but ``model`` becomes a kwarg; values are cast to float except
    ``name`` (YAML reads "3.0e3" as a string). Special keys:

    - ``hardening`` (plasticity): a ``{"law": ...}`` mapping, built into an
      IsotropicHardening; without it, ``sigma_y0``/``H`` give linear hardening.
    - ``fiber_dir`` (required for transverse_isotropic, no silent Z default):
      ``[x, y, z]`` for one direction, or ``from_input`` for the per-voxel
      ``orientations`` passed in.

    Parameters
    ----------
    cfg          : one ``materials:`` entry
    orientations : (3, Nv) fiber directions, used only for ``fiber_dir: from_input``
    """
    cfg = dict(cfg)
    model = cfg.pop("model", None)
    if model not in _MODELS:
        raise ValueError(f"unknown material model {model!r}, expected one of {list(_MODELS)}")

    fiber_dir_cfg = cfg.pop("fiber_dir", None)
    hardening_cfg = cfg.pop("hardening", None)
    kwargs = {k: (v if k == "name" else float(v)) for k, v in cfg.items()}

    if hardening_cfg is not None:
        kwargs["hardening"] = build_hardening(hardening_cfg)

    if model == "transverse_isotropic" and fiber_dir_cfg is None:
        raise ValueError(
            f"material {cfg.get('name', '')!r}: model 'transverse_isotropic' requires "
            "an explicit fiber_dir -- either a literal direction "
            "(fiber_dir: [x, y, z]) or fiber_dir: from_input for a per-voxel "
            "orientation field. There's no config-level default: silently assuming "
            "the reference axis Z = [0, 0, 1] would be easy to get wrong without "
            "noticing."
        )

    if fiber_dir_cfg is not None:
        if fiber_dir_cfg == "from_input":
            if orientations is None:
                raise ValueError(
                    f"material {cfg.get('name', '')!r} sets fiber_dir: from_input, "
                    "but no per-voxel orientation field was provided -- the geometry "
                    "file has no orientation data, or the caller didn't pass it "
                    "through to build_material(cfg, orientations=...)"
                )
            kwargs["fiber_dir"] = orientations
        else:
            kwargs["fiber_dir"] = [float(v) for v in fiber_dir_cfg]

    return _MODELS[model](**kwargs)
