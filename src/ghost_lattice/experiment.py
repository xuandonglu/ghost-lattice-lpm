"""Prepare published EBSD/HR-DIC files for measured-microstructure calculations."""

import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from .numerics import orientations as eg

HOMOLOGOUS_POINTS = {
    "CG": (
        [(250.3, 428.1), (449.6, 507.5), (332.3, 362.1), (272, 516.9), (325.2, 508.4)],
        [(114, 218), (719, 456), (354, 29), (186, 496), (339, 473)],
    ),
    "FG": (
        [
            (1322.9, 242.3),
            (1330.9, 421.9),
            (1280.3, 573.6),
            (1369.2, 717.9),
            (1571.9, 313.6),
            (1814.8, 346.7),
        ],
        [
            (69.7, 37.9),
            (79.6, 258.7),
            (25.7, 448.0),
            (130.9, 618.5),
            (362.1, 130.5),
            (644.4, 167.3),
        ],
    ),
}


def read_strain_sheet(path):
    """The published spreadsheets have one index row and one index column."""
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.worksheets[0].iter_rows(min_row=2, min_col=2, values_only=True)
        return np.array(
            [
                [float(x) if isinstance(x, (int, float)) else np.nan for x in row]
                for row in rows
            ]
        )
    finally:
        workbook.close()


def clean_strain(values):
    result = values.copy()
    valid = np.isfinite(result)
    if not valid.any():
        raise ValueError("Strain map contains no finite samples")
    bad = valid & (result > 5 * np.quantile(result[valid], 0.999))
    bad = ndimage.binary_dilation(bad, structure=np.ones((5, 5), bool))
    result[bad] = np.nan
    return result


def merge_crop_lamellae(grains, minimum_radius):
    """Use per-region EDT radii for the historical measured-window recipe.

    This differs from the later shared-boundary distance approximation in
    numerics.orientations.merge_thin. The crop recipe is kept separate.
    """
    grains = np.unique(grains, return_inverse=True)[1].reshape(grains.shape)
    for _ in range(50):
        count = int(grains.max()) + 1
        radii = np.array(
            [ndimage.distance_transform_edt(grains == k).max() for k in range(count)]
        )
        flagged = radii < minimum_radius
        if not flagged.any():
            break
        grains, changed = eg._merge_into_neighbours(grains, flagged)
        grains = np.unique(grains, return_inverse=True)[1].reshape(grains.shape)
        if not changed:
            break
    return grains


def crop_grains(
    prep, transform, dic_shape, crop_um=64.0, base_spacing_um=0.5, specimen=None
):
    """Adopt the original grain segmentation and small/thin-region merging."""
    G, step = prep["G"], float(prep["step"])
    ny, nx = dic_shape
    center = np.array([nx / 2, ny / 2, 1.0]) @ transform
    x0, y0 = np.round(center - crop_um / (2 * step)).astype(int)
    n = int(round(crop_um / step))
    if x0 < 0 or y0 < 0 or x0 + n > G.shape[1] or y0 + n > G.shape[0]:
        raise ValueError("The requested physical window exceeds the EBSD map")
    original = G[y0 : y0 + n, x0 : x0 + n]
    grains = np.unique(original, return_inverse=True)[1].reshape(n, n)
    px = crop_um / base_spacing_um / n
    grains = eg.merge_small(grains, max(10, int(round(6.0 / px**2))))
    grains = merge_crop_lamellae(grains, 2.0 / px)
    if specimen == "CG":
        # Explicit border-fragment assignment reproduces the adopted CG grid.
        # Coordinates are (row, column) in its 128-by-128, 64-um crop.
        if grains.shape != (128, 128) or crop_um != 64 or base_spacing_um != 0.5:
            raise ValueError(
                "The adopted CG border-fragment recipe requires its 64-um reference crop"
            )
        source, destination = grains[33, 127], grains[37, 123]
        if np.count_nonzero(grains == source) != 15:
            raise ValueError(
                "CG border-fragment input differs from the adopted preprocessing recipe"
            )
        grains[grains == source] = destination
        grains = np.unique(grains, return_inverse=True)[1].reshape(grains.shape)
    ids = np.array(
        [
            np.bincount(original[grains == k]).argmax()
            for k in range(int(grains.max()) + 1)
        ]
    )
    return dict(
        G=grains,
        px=px,
        sf=prep["sf"][ids],
        phi1=prep["ori"][ids, 0],
        grain_orig=ids,
        crop_um=crop_um,
        a_um=base_spacing_um,
        x0=x0,
        y0=y0,
        step=step,
    )


def prepare(raw, output, which):
    """Write new processed inputs; experimental observations are never synthesized."""
    if which not in HOMOLOGOUS_POINTS:
        raise ValueError("which must be CG or FG")
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [
        f"hea_{which}_prep.npz",
        f"hea_{which}_grains_W128t.npz",
        f"hea_{which}_experiment.npz",
        f"hea_{which}_processing.json",
    ]
    if any((output / name).exists() for name in names):
        raise FileExistsError(
            "Use an output folder without existing inputs for this specimen"
        )
    maps, cleaned = {}, {}
    for letter, load in zip("ABC", ("0.75", "1.50", "2.25")):
        path = raw / f"Fig.5({letter}){which}-HEA_{load}%_effective shear strain.xlsx"
        maps[load] = read_strain_sheet(path)
        cleaned[load] = clean_strain(maps[load])
    ebsd = eg.read_ctf(raw / f"Fig.3{which}-HEA_HRDIC.ctf")
    grains = eg.merge_small(eg.segment(ebsd["euler"], ebsd["phase"], tol=8.0), 10)
    ori = eg.grain_mean_quat(grains, ebsd["euler"])
    sf = eg.schmid_fcc(ori, (1.0, 0.0, 0.0))
    boundary = np.zeros_like(grains, bool)
    boundary[:, :-1] |= grains[:, :-1] != grains[:, 1:]
    boundary[:, 1:] |= grains[:, :-1] != grains[:, 1:]
    boundary[:-1] |= grains[:-1] != grains[1:]
    boundary[1:] |= grains[:-1] != grains[1:]
    distance = ndimage.distance_transform_edt(~boundary) * ebsd["step"]
    prep = dict(
        G=grains, step=ebsd["step"], euler=ebsd["euler"], sf=sf, dist=distance, ori=ori
    )
    target, source = map(np.asarray, HOMOLOGOUS_POINTS[which])
    transform = np.linalg.lstsq(
        np.column_stack([source, np.ones(len(source))]), target, rcond=None
    )[0]
    crop = crop_grains(prep, transform, maps["0.75"].shape, specimen=which)
    yy, xx = np.indices(maps["0.75"].shape)
    coords = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)]) @ transform
    ix = np.clip(np.round(coords[:, 0]).astype(int), 0, grains.shape[1] - 1)
    iy = np.clip(np.round(coords[:, 1]).astype(int), 0, grains.shape[0] - 1)
    np.savez_compressed(output / names[0], **prep)
    np.savez_compressed(output / names[1], **crop)
    np.savez_compressed(
        output / names[2],
        **{f"raw_{k}": v for k, v in maps.items()},
        **{f"clean_{k}": v for k, v in cleaned.items()},
        transform=transform,
        grain=grains[iy, ix].reshape(xx.shape),
        distance_um=distance[iy, ix].reshape(xx.shape),
    )
    summary = dict(
        specimen=which,
        source_doi="10.5281/zenodo.14532401",
        source_license="CC-BY-4.0",
        segmentation_degrees=8,
        minimum_pixels=10,
        crop_um=64,
        base_spacing_um=0.5,
        grains=int(crop["G"].max()) + 1,
        orientation="Euler angles at each segmented grain's most central pixel",
        cleaning="Mask values above five times the 99.9th percentile and dilate the mask with a 5x5 stencil",
        crop_radius_measure="Per-grain Euclidean distance transform on the crop",
        border_fragment_recipe="CG: merge the 15-pixel region at (33,127) into the region at (37,123); FG: none",
        outputs=names[:3],
    )
    (output / names[3]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
