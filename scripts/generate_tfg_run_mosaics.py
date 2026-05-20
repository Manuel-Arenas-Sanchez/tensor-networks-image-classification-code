from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tfg_run_mosaics"

RUNS = [
    ROOT / "runs_server" / "20260507-085758-ChemEq_exp1",
    ROOT / "runs_server" / "20260507-130407-ChemEq_exp2",
    ROOT / "runs_server" / "20260508-140030-ChemEq_exp3",
    ROOT / "runs_server" / "20260511-095831-ChemEq_ls005_seed42",
    ROOT / "runs_server" / "20260511-110143-ChemEq_exp6",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_mosaic(
        [run / "loss_curve.png" for run in RUNS],
        OUT_DIR / "tfg_loss_curves_mosaic.png",
    )
    write_mosaic(
        [run / "diagnostics" / "confusion_matrix.png" for run in RUNS],
        OUT_DIR / "tfg_confusion_matrices_mosaic.png",
    )
    print(f"Wrote mosaics to {OUT_DIR}")


def write_mosaic(paths: list[Path], output_path: Path) -> None:
    if len(paths) != 5:
        raise ValueError("This mosaic layout expects exactly five images.")

    images = [trim_white_border(Image.open(path).convert("RGB")) for path in paths]
    top_images = images[:3]
    bottom_images = images[3:]

    top_cell_width = 1600
    top_cell_height = round(top_cell_width / top_images[0].width * top_images[0].height)
    output_width = top_cell_width * 3

    bottom_cell_width = output_width // 2
    bottom_cell_height = round(bottom_cell_width / bottom_images[0].width * bottom_images[0].height)
    output_height = top_cell_height + bottom_cell_height

    canvas = Image.new("RGB", (output_width, output_height), "white")

    for index, image in enumerate(top_images):
        resized = image.resize((top_cell_width, top_cell_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (index * top_cell_width, 0))

    for index, image in enumerate(bottom_images):
        resized = image.resize((bottom_cell_width, bottom_cell_height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (index * bottom_cell_width, top_cell_height))

    canvas.save(output_path, dpi=(300, 300))


def trim_white_border(image: Image.Image) -> Image.Image:
    white = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, white).convert("L")
    mask = difference.point(lambda value: 255 if value > 6 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


if __name__ == "__main__":
    main()
