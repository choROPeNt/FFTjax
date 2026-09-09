"""Build a ConstitutiveModel from a config dict, keyed by a "model" string --
so YAML-driven scripts can pick the material model per phase instead of a
script hardcoding one class for all phases (e.g. mixed isotropic/transversely
isotropic phases in one materials: list)."""

from materialmodels.base import ConstitutiveModel
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from materialmodels.inelastic.plasticity_drucker_prager import DruckerPrager
from materialmodels.inelastic.plasticity_j2 import J2Plasticity

_MODELS = {
    "isotropic_elastic":    LinearElasticIsotropic,
    "transverse_isotropic": TransverseIsotropic,
    "j2_plasticity":        J2Plasticity,
    "drucker_prager":       DruckerPrager,
}


def build_material(cfg: dict, orientations=None) -> ConstitutiveModel:
    """
    cfg: one entry of a YAML materials: list, e.g.
    {"model": "isotropic_elastic", "E": 3.0e3, "nu": 0.35, "name": "epoxy matrix"}
    -- "model" selects the class, every other key is forwarded as a kwarg
    ("name" passed through as-is, everything else cast to float -- YAML's
    float regex doesn't recognize exponents without an explicit sign, e.g.
    "3.0e3" loads as a str, not 3000.0).

    ``fiber_dir`` (transverse_isotropic only -- see TransverseIsotropic's
    docstring) is handled specially, before the blanket float-cast, since it
    isn't a scalar, and is REQUIRED for this model (unlike
    TransverseIsotropic itself, which defaults to the reference axis Z when
    constructed directly) -- a config silently relying on that default is
    easy to get wrong (wrong/no orientation) without ever noticing, so
    config-driven construction demands it be stated explicitly:

    - a literal ``fiber_dir: [x, y, z]`` in the config -> one fixed global
      direction for this phase, forwarded as a plain list.
    - ``fiber_dir: from_input`` -> this phase's fibre direction varies per
      voxel, taken from the geometry file's own orientation field (e.g.
      utils.io.reader.SimulationReader/read_vtu's ``orientations`` output,
      (3, Nv)) -- the caller must pass that array in as ``orientations``, or
      building this material raises.
    - omitted entirely -> raises immediately, naming the material.

    Parameters
    ----------
    cfg          : one materials: list entry
    orientations : (3, Nv) per-voxel fibre direction field, or None -- only
                   consulted when cfg sets ``fiber_dir: from_input``; every
                   other config ignores it, so it's safe to pass the same
                   array to every build_material call regardless of model.
    """
    cfg = dict(cfg)
    model = cfg.pop("model", None)
    if model not in _MODELS:
        raise ValueError(f"unknown material model {model!r}, expected one of {list(_MODELS)}")

    fiber_dir_cfg = cfg.pop("fiber_dir", None)
    kwargs = {k: (v if k == "name" else float(v)) for k, v in cfg.items()}

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
