from __future__ import annotations

import math
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ChemistryLabApparatus-25"
OUT_DIR = ROOT / "tfg_dataset_figures"

CROP_MARGIN = 0.10
INPUT_SIZE = 224
PAD_FILL = (124, 116, 104)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BOX_COLORS = [
    (216, 45, 45),
    (32, 102, 191),
    (39, 137, 85),
    (201, 112, 22),
    (133, 72, 174),
    (31, 145, 160),
    (196, 52, 128),
    (93, 115, 29),
]

CLASS_SHORT_NAMES = {
    "Round_Bottom_Flask_Borosilicate_Glass_1_Neck": "Round flask 1-neck",
    "Round_Bottom_Flask_Borosilicate_Glass_2_Neck": "Round flask 2-neck",
    "Round_Bottom_Flask_Borosilicate_Glass_3_Neck": "Round flask 3-neck",
    "Porcelain_Mortar Pestle": "Mortar and pestle",
    "Mechanical_Balance_Scale": "Balance scale",
    "Nessler_Reagent_Bottle": "Nessler bottle",
    "Precision_Weight_Scale": "Precision scale",
    "TestTube_Holder": "Test tube holder",
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    split: str
    object_index: int
    class_id: int
    class_name: str
    raw_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    split: str
    samples: tuple[Sample, ...]
    image_size: tuple[int, int]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(DATA_DIR / "data.yaml")
    samples = build_samples(DATA_DIR, class_names)
    records = build_image_records(samples)

    box_records = select_image_records(records, n=9)
    crop_samples = select_crop_samples(samples, n=16)

    write_box_mosaic(
        box_records,
        OUT_DIR / "tfg_dataset_yolo_boxes_square.png",
        columns=3,
        rows=3,
        canvas_size=(2400, 2400),
        title="Ejemplos de cajas YOLO sobre imagen original",
    )
    write_crop_mosaic(
        crop_samples,
        OUT_DIR / "tfg_dataset_classifier_crops_square.png",
        columns=4,
        rows=4,
        canvas_size=(2400, 2400),
        title="Crops usados como entrada del clasificador",
    )
    write_combined_mosaic(
        box_records[:6],
        crop_samples[:8],
        OUT_DIR / "tfg_dataset_boxes_and_crops_square.png",
    )

    print(f"Wrote figures to {OUT_DIR}")


def load_class_names(path: Path) -> list[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            names = ast.literal_eval(stripped.removeprefix("names:").strip())
            if isinstance(names, dict):
                return [names[index] for index in sorted(names)]
            return list(names)
    raise ValueError(f"Could not find class names in {path}")


def build_samples(root_dir: Path, class_names: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for split in ("train", "valid", "test"):
        images_dir = root_dir / split / "images"
        labels_dir = root_dir / split / "labels"

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image_path(images_dir, label_path.stem)
            with Image.open(image_path) as image:
                image_size = image.size

            object_index = 0
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped:
                    continue

                class_id, bbox_normalized = parse_yolo_line(stripped, label_path, line_number)
                raw_box = yolo_to_pixel_box(bbox_normalized, image_size, margin=0.0)
                crop_box = yolo_to_pixel_box(bbox_normalized, image_size, margin=CROP_MARGIN)

                samples.append(
                    Sample(
                        image_path=image_path,
                        label_path=label_path,
                        split=split,
                        object_index=object_index,
                        class_id=class_id,
                        class_name=class_names[class_id],
                        raw_box=raw_box,
                        crop_box=crop_box,
                        image_size=image_size,
                    )
                )
                object_index += 1
    return samples


def build_image_records(samples: list[Sample]) -> list[ImageRecord]:
    grouped: dict[Path, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.image_path].append(sample)

    records = [
        ImageRecord(
            image_path=image_path,
            split=image_samples[0].split,
            samples=tuple(image_samples),
            image_size=image_samples[0].image_size,
        )
        for image_path, image_samples in grouped.items()
    ]
    return sorted(records, key=lambda record: (record.split, str(record.image_path)))


def find_image_path(images_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {stem} in {images_dir}")


def parse_yolo_line(
    line: str,
    label_path: Path,
    line_number: int,
) -> tuple[int, tuple[float, float, float, float]]:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO row in {label_path}:{line_number}")

    class_id = int(parts[0])
    if len(parts) == 5:
        x_center, y_center, width, height = (float(value) for value in parts[1:])
        return class_id, (x_center, y_center, width, height)

    coordinates = [float(value) for value in parts[1:]]
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    width = right - left
    height = bottom - top
    return class_id, (left + width / 2, top + height / 2, width, height)


def yolo_to_pixel_box(
    bbox_normalized: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    margin: float,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x_center, y_center, width, height = bbox_normalized

    box_width = width * image_width
    box_height = height * image_height

    left = x_center * image_width - box_width / 2
    top = y_center * image_height - box_height / 2
    right = x_center * image_width + box_width / 2
    bottom = y_center * image_height + box_height / 2

    left -= box_width * margin
    right += box_width * margin
    top -= box_height * margin
    bottom += box_height * margin

    left_i = max(0, math.floor(left))
    top_i = max(0, math.floor(top))
    right_i = min(image_width, math.ceil(right))
    bottom_i = min(image_height, math.ceil(bottom))

    if right_i <= left_i:
        right_i = min(image_width, left_i + 1)
    if bottom_i <= top_i:
        bottom_i = min(image_height, top_i + 1)
    return left_i, top_i, right_i, bottom_i


def select_image_records(records: list[ImageRecord], n: int) -> list[ImageRecord]:
    candidates = [
        record for record in records
        if 1 <= len(record.samples) <= 5 and max_box_area_ratio(record) >= 0.015
    ]
    if len(candidates) < n:
        candidates = [record for record in records if len(record.samples) >= 1]

    selected: list[ImageRecord] = []
    covered_classes: set[int] = set()
    target_per_split = {"train": 3, "valid": 3, "test": 3}

    for split, target in target_per_split.items():
        while sum(record.split == split for record in selected) < target:
            remaining = [
                record for record in candidates
                if record not in selected and record.split == split
            ]
            if not remaining:
                break
            best = max(
                remaining,
                key=lambda record: image_record_score(record, covered_classes),
            )
            selected.append(best)
            covered_classes.update(sample.class_id for sample in best.samples)

    if len(selected) < n:
        remaining = [record for record in candidates if record not in selected]
        selected.extend(
            sorted(
                remaining,
                key=lambda record: image_record_score(record, covered_classes),
                reverse=True,
            )[: n - len(selected)]
        )

    return selected[:n]


def image_record_score(record: ImageRecord, covered_classes: set[int]) -> float:
    class_ids = {sample.class_id for sample in record.samples}
    new_classes = len(class_ids - covered_classes)
    object_count = len(record.samples)
    area = max_box_area_ratio(record)
    object_bonus = 0.4 if 2 <= object_count <= 4 else 0.0
    crowd_penalty = max(0, object_count - 4) * 0.8
    return new_classes * 10.0 + area * 3.0 + object_bonus - crowd_penalty


def max_box_area_ratio(record: ImageRecord) -> float:
    image_width, image_height = record.image_size
    image_area = image_width * image_height
    return max(
        ((right - left) * (bottom - top)) / image_area
        for left, top, right, bottom in (sample.raw_box for sample in record.samples)
    )


def select_crop_samples(samples: list[Sample], n: int) -> list[Sample]:
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_id].append(sample)

    selected: list[Sample] = []
    class_order = [
        0, 1, 2, 4, 5, 6, 7, 10,
        13, 14, 15, 16, 17, 20, 21, 22,
        23, 24, 8, 9, 11, 12, 18, 19,
    ]
    for class_id in class_order:
        if len(selected) >= n:
            break
        if class_id not in by_class:
            continue
        selected.append(max(by_class[class_id], key=crop_sample_score))

    if len(selected) < n:
        remaining = [sample for sample in samples if sample not in selected]
        selected.extend(sorted(remaining, key=crop_sample_score, reverse=True)[: n - len(selected)])
    return selected[:n]


def crop_sample_score(sample: Sample) -> float:
    image_width, image_height = sample.image_size
    left, top, right, bottom = sample.raw_box
    width = right - left
    height = bottom - top
    area_ratio = (width * height) / (image_width * image_height)
    aspect_ratio = max(width / max(height, 1), height / max(width, 1))

    area_score = 1.0 - abs(math.log(max(area_ratio, 1e-6) / 0.08))
    aspect_score = -abs(math.log(max(aspect_ratio, 1e-6) / 1.6))
    edge_distance = min(left, top, image_width - right, image_height - bottom)
    edge_score = min(max(edge_distance, 0) / (0.08 * min(image_width, image_height)), 1.0)
    crop_left, crop_top, crop_right, crop_bottom = sample.crop_box
    crop_edge_penalty = sum(
        [
            crop_left <= 0,
            crop_top <= 0,
            crop_right >= image_width,
            crop_bottom >= image_height,
        ]
    )
    split_bonus = {"train": 0.2, "valid": 0.1, "test": 0.0}[sample.split]
    return area_score * 1.6 + aspect_score + edge_score * 2.2 - crop_edge_penalty * 1.3 + split_bonus


def write_box_mosaic(
    records: list[ImageRecord],
    path: Path,
    *,
    columns: int,
    rows: int,
    canvas_size: tuple[int, int],
    title: str,
) -> None:
    canvas = Image.new("RGB", canvas_size, "white")
    fonts = load_fonts()
    cell_width = canvas_size[0] // columns
    cell_height = canvas_size[1] // rows

    for index, record in enumerate(records[: columns * rows]):
        row = index // columns
        col = index % columns
        x = col * cell_width
        y = row * cell_height
        tile = render_box_tile(record, (cell_width, cell_height), fonts)
        canvas.paste(tile, (x, y))

    canvas.save(path, dpi=(300, 300))


def write_crop_mosaic(
    samples: list[Sample],
    path: Path,
    *,
    columns: int,
    rows: int,
    canvas_size: tuple[int, int],
    title: str,
) -> None:
    canvas = Image.new("RGB", canvas_size, "white")
    fonts = load_fonts()
    cell_width = canvas_size[0] // columns
    cell_height = canvas_size[1] // rows

    for index, sample in enumerate(samples[: columns * rows]):
        row = index // columns
        col = index % columns
        x = col * cell_width
        y = row * cell_height
        tile = render_crop_tile(sample, (cell_width, cell_height), fonts)
        canvas.paste(tile, (x, y))

    canvas.save(path, dpi=(300, 300))


def write_combined_mosaic(
    box_records: list[ImageRecord],
    crop_samples: list[Sample],
    path: Path,
) -> None:
    canvas_size = (2400, 2400)
    canvas = Image.new("RGB", canvas_size, "white")
    fonts = load_fonts()
    box_columns = 3
    box_rows = 2
    box_cell_width = canvas_size[0] // box_columns
    box_cell_height = canvas_size[1] // 2 // box_rows

    for index, record in enumerate(box_records[: box_columns * box_rows]):
        row = index // box_columns
        col = index % box_columns
        x = col * box_cell_width
        y = row * box_cell_height
        tile = render_box_tile(record, (box_cell_width, box_cell_height), fonts)
        canvas.paste(tile, (x, y))

    crop_columns = 4
    crop_rows = 2
    crop_cell_width = canvas_size[0] // crop_columns
    crop_cell_height = canvas_size[1] // 2 // crop_rows

    for index, sample in enumerate(crop_samples[: crop_columns * crop_rows]):
        row = index // crop_columns
        col = index % crop_columns
        x = col * crop_cell_width
        y = canvas_size[1] // 2 + row * crop_cell_height
        tile = render_crop_tile(sample, (crop_cell_width, crop_cell_height), fonts)
        canvas.paste(tile, (x, y))

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_panel_label(draw, "Cajas YOLO", (18, 18), fonts)
    draw_panel_label(draw, "Crops del clasificador", (18, canvas_size[1] // 2 + 18), fonts)
    canvas.save(path, dpi=(300, 300))


def render_box_tile(
    record: ImageRecord,
    tile_size: tuple[int, int],
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    with Image.open(record.image_path) as image:
        image = image.convert("RGB")
        tile, scale, offset = cover_image(image, tile_size)

    draw = ImageDraw.Draw(tile, "RGBA")

    for sample_index, sample in enumerate(record.samples):
        color = BOX_COLORS[(sample.class_id + sample_index) % len(BOX_COLORS)]
        left, top, right, bottom = sample.raw_box
        scaled_box = (
            round(offset[0] + left * scale),
            round(offset[1] + top * scale),
            round(offset[0] + right * scale),
            round(offset[1] + bottom * scale),
        )
        width = max(4, round(5 * scale))
        for inset in range(width):
            draw.rectangle(
                (
                    scaled_box[0] - inset,
                    scaled_box[1] - inset,
                    scaled_box[2] + inset,
                    scaled_box[3] + inset,
                ),
                outline=color + (245,),
            )
        label = f"{sample.class_id:02d}"
        label_box = text_bbox(draw, label, fonts["small_bold"])
        label_width = label_box[2] - label_box[0] + 16
        label_height = label_box[3] - label_box[1] + 10
        label_x = min(max(scaled_box[0], 6), max(6, tile_size[0] - label_width - 6))
        label_y = min(
            max(scaled_box[1] - label_height - 2, 6),
            max(6, tile_size[1] - label_height - 6),
        )
        draw.rounded_rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            radius=4,
            fill=color + (230,),
        )
        draw.text((label_x + 8, label_y + 4), label, fill=(255, 255, 255, 255), font=fonts["small_bold"])

    split_text = record.split.upper()
    split_box = text_bbox(draw, split_text, fonts["small_bold"])
    split_w = split_box[2] - split_box[0] + 20
    split_h = split_box[3] - split_box[1] + 12
    draw.rounded_rectangle(
        (18, 18, 18 + split_w, 18 + split_h),
        radius=4,
        fill=(30, 35, 40, 220),
    )
    draw.text((28, 23), split_text, fill=(255, 255, 255, 255), font=fonts["small_bold"])
    return tile


def render_crop_tile(
    sample: Sample,
    tile_size: tuple[int, int],
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    with Image.open(sample.image_path) as image:
        crop = image.convert("RGB").crop(sample.crop_box)
    crop = resize_with_padding(crop, INPUT_SIZE, PAD_FILL)
    tile, _, _ = cover_image(crop, tile_size)
    draw = ImageDraw.Draw(tile, "RGBA")

    name = short_class_name(sample.class_name)
    label = f"{sample.class_id:02d}  {name}"
    lines = wrap_text(label, fonts["label"], tile_size[0] - 28)
    label_text = lines[0]
    if len(lines) > 1:
        label_text = f"{label_text}..."
    label_box = text_bbox(draw, label_text, fonts["label"])
    label_width = label_box[2] - label_box[0] + 24
    label_height = label_box[3] - label_box[1] + 16
    label_x = max(0, (tile_size[0] - label_width) // 2)
    label_y = tile_size[1] - label_height - 12
    draw.rounded_rectangle(
        (label_x, label_y, label_x + label_width, label_y + label_height),
        radius=5,
        fill=(30, 35, 40, 210),
    )
    draw.text(
        (label_x + 12, label_y + 7),
        label_text,
        fill=(255, 255, 255, 255),
        font=fonts["label"],
    )
    return tile


def cover_image(
    image: Image.Image,
    target_size: tuple[int, int],
) -> tuple[Image.Image, float, tuple[int, int]]:
    target_width, target_height = target_size
    scale = max(target_width / image.width, target_height / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    offset = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    canvas = Image.new("RGB", target_size, "white")
    canvas.paste(resized, offset)
    return canvas, scale, offset


def resize_with_padding(image: Image.Image, size: int, fill: tuple[int, int, int]) -> Image.Image:
    scale = min(size / image.width, size / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def load_fonts() -> dict[str, ImageFont.ImageFont]:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    regular_path = next((path for path in candidates if path.is_file()), None)
    bold_path = next((path for path in bold_candidates if path.is_file()), regular_path)
    if regular_path is None:
        default = ImageFont.load_default()
        return {"title": default, "label": default, "small_bold": default}
    return {
        "title": ImageFont.truetype(str(bold_path), 44),
        "label": ImageFont.truetype(str(regular_path), 26),
        "small_bold": ImageFont.truetype(str(bold_path), 24),
    }


def short_class_name(name: str) -> str:
    if name in CLASS_SHORT_NAMES:
        return CLASS_SHORT_NAMES[name]
    return name.replace("_", " ")


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_bbox(draw, candidate, font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def draw_panel_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    x, y = position
    box = text_bbox(draw, text, fonts["title"])
    width = box[2] - box[0] + 34
    height = box[3] - box[1] + 22
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=6,
        fill=(30, 35, 40, 225),
    )
    draw.text((x + 17, y + 9), text, fill=(255, 255, 255, 255), font=fonts["title"])


if __name__ == "__main__":
    main()
