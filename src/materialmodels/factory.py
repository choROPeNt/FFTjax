"""Build a ConstitutiveModel from a config dict, keyed by a "model" string --
so YAML-driven scripts can pick the material model per phase instead of a
script hardcoding one class for all phases (e.g. mixed isotropic/transversely
isotropic phases in one materials: list)."""

from materialmodels.base import ConstitutiveModel
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transversely_isotropic import TransverseIsotropicFibre

_MODELS = {
    "isotropic_elastic":    LinearElasticIsotropic,
    "transverse_isotropic": TransverseIsotropicFibre,
}


def build_material(cfg: dict) -> ConstitutiveModel:
    """cfg: one entry of a YAML materials: list, e.g.
    {"model": "isotropic_elastic", "E": 3.0e3, "nu": 0.35, "name": "epoxy matrix"}
    -- "model" selects the class, every other key is forwarded as a kwarg
    ("name" passed through as-is, everything else cast to float -- YAML's
    float regex doesn't recognize exponents without an explicit sign, e.g.
    "3.0e3" loads as a str, not 3000.0)."""
    cfg = dict(cfg)
    model = cfg.pop("model", None)
    if model not in _MODELS:
        raise ValueError(f"unknown material model {model!r}, expected one of {list(_MODELS)}")
    kwargs = {k: (v if k == "name" else float(v)) for k, v in cfg.items()}
    return _MODELS[model](**kwargs)
