"""Подготовка датасета SIVED к обучению YOLO Detect.

Исходный датасет SIVED (https://github.com/.../SIVED) поставляется с разметкой
в формате DOTA — объект описывается четырьмя точками повёрнутого
прямоугольника:

    x1 y1 x2 y2 x3 y3 x4 y4 Vehicle 0

Для обучения обычной модели YOLO Detect такая разметка приводится к
горизонтальному (неориентированному) bounding box:

    0 x_center y_center width height

Скрипт копирует изображения и конвертирует разметку, создавая структуру:

    dataset/
      images/{train,valid,test}/*.jpg
      labels/{train,valid,test}/*.txt

Пример запуска:

    python scripts/prepare_dataset.py --source D:/SIVED-main/SIVED-main/SIVED
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CLASS_NAME = "Vehicle"


def resolve_source(source: Path) -> Path:
    """Находит корень датасета внутри указанного пути."""
    candidates = [source, source / "SIVED"]

    for candidate in candidates:
        if (candidate / "ImageSets" / "labelTxt").is_dir():
            return candidate

    raise SystemExit(
        f"В {source} не найдена папка ImageSets/labelTxt. "
        "Укажите путь к корню датасета SIVED."
    )


def find_image(split: str, stem: str, images_root: Path) -> Path:
    for extension in IMAGE_EXTENSIONS:
        image_path = images_root / split / f"{stem}{extension}"

        if image_path.exists():
            return image_path

    raise FileNotFoundError(
        f"Не найдено изображение для labels/{split}/{stem}.txt"
    )


def convert_dota_line(
    parts: list[str],
    image_width: int,
    image_height: int,
    file_path: Path,
    line_number: int,
) -> str:
    """Преобразует DOTA/OBB-разметку в стандартный YOLO HBB."""
    if len(parts) != 10:
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            f"ожидалось 10 значений DOTA, найдено {len(parts)}"
        )

    try:
        coordinates = list(map(float, parts[:8]))
    except ValueError as error:
        raise ValueError(
            f"{file_path}, строка {line_number}: ошибка чтения координат"
        ) from error

    if parts[8] != CLASS_NAME:
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            f"неизвестный класс {parts[8]!r}"
        )

    xs = coordinates[0::2]
    ys = coordinates[1::2]

    xmin = max(0.0, min(xs))
    xmax = min(float(image_width), max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(float(image_height), max(ys))

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0 or box_height <= 0:
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            "прямоугольник имеет нулевой или отрицательный размер"
        )

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height

    return (
        f"0 "
        f"{x_center:.10f} "
        f"{y_center:.10f} "
        f"{box_width / image_width:.10f} "
        f"{box_height / image_height:.10f}"
    )


def process_split(
    split: str,
    source: Path,
    dataset_root: Path,
    copy_images: bool,
) -> tuple[int, int, int]:
    """Копирует изображения и конвертирует разметку одного сплита."""
    source_images = source / "ImageSets" / "images" / split
    source_labels = source / "ImageSets" / "labelTxt" / split

    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(
            f"Не найдены папки сплита {split}: {source_images} / {source_labels}"
        )

    target_images = dataset_root / "images" / split
    target_labels = dataset_root / "labels" / split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    label_files = sorted(source_labels.glob("*.txt"))
    images_written = 0
    objects_written = 0

    for label_path in label_files:
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            print(f"ПРЕДУПРЕЖДЕНИЕ: пустой файл {label_path.name}")
            continue

        image_path = find_image(split, label_path.stem, source / "ImageSets" / "images")

        if copy_images:
            shutil.copy2(image_path, target_images / image_path.name)
            images_written += 1

        converted_lines: list[str] = []

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            # Файл уже в формате YOLO Detect — оставляем как есть.
            if len(parts) == 5:
                converted_lines.append(line)
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise RuntimeError(f"Не удалось открыть изображение: {image_path}")

            image_height, image_width = image.shape[:2]

            converted_lines.append(
                convert_dota_line(
                    parts=parts,
                    image_width=image_width,
                    image_height=image_height,
                    file_path=label_path,
                    line_number=line_number,
                )
            )

        (target_labels / label_path.name).write_text(
            "\n".join(converted_lines) + "\n",
            encoding="utf-8",
        )
        objects_written += len(converted_lines)

    return images_written, len(label_files), objects_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка датасета SIVED (DOTA) к формату YOLO Detect."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Путь к исходному датасету SIVED (папка с ImageSets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="Куда сложить подготовленный датасет (по умолчанию ./dataset).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Не копировать изображения, только конвертировать разметку.",
    )
    args = parser.parse_args()

    source = resolve_source(args.source.resolve())
    dataset_root = args.output.resolve()

    print("Источник:", source)
    print("Результат:", dataset_root)
    print()

    totals = {"images": 0, "labels": 0, "objects": 0}

    for split in SPLITS:
        images_written, labels_written, objects = process_split(
            split=split,
            source=source,
            dataset_root=dataset_root,
            copy_images=not args.no_images,
        )

        totals["images"] += images_written
        totals["labels"] += labels_written
        totals["objects"] += objects

        print(
            f"{split:>5}: изображений — {images_written}, "
            f"аннотаций — {labels_written}, объектов — {objects}"
        )

    print()
    print(
        f"Итого: изображений — {totals['images']}, "
        f"аннотаций — {totals['labels']}, "
        f"объектов — {totals['objects']}"
    )
    print()
    print("Готово. Можно запускать обучение: python train.py")


if __name__ == "__main__":
    main()
