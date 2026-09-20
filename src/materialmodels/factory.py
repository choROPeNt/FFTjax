"""Build a ConstitutiveModel from a config dict, keyed by a "model" string --
so YAML-driven scripts can pick the material model per phase instead of a
script hardcoding one class for all phases (e.g. mixed isotropic/transversely
isotropic phases in one materials: list)."""

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
    cfg: one entry of a YAML materials: list, e.g.
    {"model": "isotropic_elastic", "E": 3.0e3, "nu": 0.35, "name": "epoxy matrix"}
    -- "model" selects the class, every other key is forwarded as a kwarg
    ("name" passed through as-is, everything else cast to float -- YAML's
    float regex doesn't recognize exponents without an explicit sign, e.g.
    "3.0e3" loads as a str, not 3000.0).

    "phasefield_isotropic" (PhaseFieldIsotropic -- autodiff Amor-split/AT2
    tangent, see materialmodels.phasefield.isotropic) needs no special-casing
    here: its one non-optional extra parameter, ``Gc``, is just another
    kwarg the blanket float-cast forwards -- but it IS required (the class
    itself has no default for it), so a config that omits it fails
    immediately with a missing-argument error naming this material, not
    later inside a fracture solve.

    ``hardening`` (the plasticity models only -- see
    materialmodels.inelastic.hardening) is handled specially for the same
    reason as ``fiber_dir``: it is a nested mapping, not a scalar, so the
    blanket float-cast would raise on it. It is forwarded as a built
    IsotropicHardening:

        hardening:
          law: piecewise_linear
          table:                  # (plastic_strain, yield_stress)
            - [0.000, 56.1]
            - [0.020, 70.0]

    Omitting it leaves the models' original ``sigma_y0``/``H`` linear
    hardening, which the blanket cast forwards unchanged.

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
