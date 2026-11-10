"""Data domains for the 2D protocol. Each domain exposes:

    domain.n_slices() -> int
    domain.load_image(z) -> (H, W) uint8
    domain.load_label(z) -> (H, W) integer instance ids (raw, before 2D relabel)
    domain.splits(scheme) -> {"train": [...], "val": [...], "test": [...]}  (slice indices)

Test slices of mouse_30um in the "interim" scheme are adjacent
to the validation block; results under that scheme must be reported with this caveat.
"""
from emseg.eval.domains.mouse30um import Mouse30um
from emseg.eval.domains.snemi import Snemi
from emseg.eval.domains.cremi import Cremi

DOMAINS = {"mouse30um": Mouse30um, "snemi": Snemi, "cremi": Cremi}


def get_domain(name: str, **kwargs):
    if name not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; choose from {sorted(DOMAINS)}")
    return DOMAINS[name](**kwargs)
