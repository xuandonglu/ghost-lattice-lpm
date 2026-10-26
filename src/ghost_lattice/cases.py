"""Physical case definitions, independent of output paths and machine configuration."""

from dataclasses import asdict, dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    geometry: str = "planar"
    law: str = "j2"
    size: float = 64.0
    grains: int = 16
    spacing: float = 1.0
    band: float = 6.0
    window_margin: float = 8.0
    increment: float = 0.02
    maximum_load: float = 3.0
    checkpoints: tuple = (1.0, 3.0)
    seed: int = 1
    yield_contrast: float = 0.0
    measured_map: str = ""
    unit: str = "model length"
    reverse: bool = False
    description: str = ""

    def as_dict(self):
        return asdict(self)

    def loads(self):
        count = round(self.maximum_load / self.increment)
        forward = np.arange(1, count + 1) * self.increment
        if self.reverse:
            return np.r_[
                forward,
                self.maximum_load - np.arange(1, 2 * count + 1) * self.increment,
            ]
        return forward


CASES = {
    "demo": Case(
        "demo",
        size=16,
        grains=4,
        band=3,
        window_margin=4,
        increment=0.1,
        yield_contrast=0.4,
        description="Small demonstration; separate from the manuscript discretizations.",
    ),
    "homogeneous": Case(
        "homogeneous", description="Representative homogeneous 16-grain plate."
    ),
    "heterogeneous": Case("heterogeneous", spacing=0.05, yield_contrast=0.4),
    "reversal": Case(
        "reversal",
        spacing=0.1,
        yield_contrast=0.4,
        reverse=True,
        checkpoints=(0.0, -3.0),
    ),
    "assembly_control": Case(
        "assembly_control",
        yield_contrast=0.4,
        reverse=True,
        checkpoints=(1.0, 3.0, 0.0, -3.0),
    ),
    "planar_cp": Case(
        "planar_cp",
        law="cp",
        spacing=0.25,
        increment=0.04,
        maximum_load=2,
        checkpoints=(2.0,),
    ),
    "anisotropic": Case(
        "anisotropic",
        law="elastic",
        spacing=0.25,
        increment=1,
        maximum_load=1,
        checkpoints=(1.0,),
    ),
    "hole": Case("hole", spacing=0.125, maximum_load=2, checkpoints=(2.0,)),
    "columnar_cp": Case(
        "columnar_cp",
        geometry="columnar",
        law="cp",
        size=40,
        grains=6,
        band=5,
        window_margin=7,
        increment=0.04,
        maximum_load=2,
        checkpoints=(2.0,),
    ),
    "equiaxed_homogeneous": Case(
        "equiaxed_homogeneous",
        geometry="equiaxed",
        size=24,
        grains=8,
        band=4,
        window_margin=6,
        checkpoints=(3.0,),
    ),
    "equiaxed_j2": Case(
        "equiaxed_j2",
        geometry="equiaxed",
        size=24,
        grains=8,
        spacing=0.5,
        band=4,
        window_margin=6,
        yield_contrast=0.4,
    ),
    "equiaxed_cp": Case(
        "equiaxed_cp",
        geometry="equiaxed",
        law="cp",
        size=24,
        grains=8,
        spacing=0.5,
        band=4,
        window_margin=6,
        increment=0.05,
        maximum_load=2,
        checkpoints=(2.0,),
    ),
}
for which in ("coarse", "fine"):
    for law in ("j2", "cp"):
        key = f"{which}_{law}"
        CASES[key] = Case(
            key,
            geometry="measured",
            law=law,
            size=64,
            grains=131 if which == "coarse" else 272,
            spacing=0.05 if law == "j2" else 0.1,
            band=2,
            window_margin=3,
            increment=0.25,
            maximum_load=16,
            checkpoints=(8.0, 12.0, 16.0),
            measured_map=which,
            unit="micrometre",
        )
for law in ("static", "hallpetch", "slip_sigma3", "slip_all"):
    key = f"coarse_{law}"
    CASES[key] = replace(CASES["coarse_cp"], name=key, law=law)
CASES["fine_j2_supplement"] = replace(
    CASES["fine_j2"], name="fine_j2_supplement", spacing=0.25
)


def get_case(name, seed=None):
    case = CASES[name]
    return replace(case, seed=seed) if seed is not None else case
