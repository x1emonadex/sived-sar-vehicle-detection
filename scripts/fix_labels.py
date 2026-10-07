"""Нормализация и проверка разметки датасета.

Скрипт приводит все аннотации к формату YOLO Detect:
  * файл с 5 значениями (0 x_center y_center width height) считается готовым;
  * файл с 10 значениями (DOTA/OBB) конвертируется в горизонтальный bounding box;
  * исходная DOTA-разметка сохраняется в dataset/labels_dota_backup;
  * проверяются диапазоны значений и соответствие числа изображений и аннотаций;
  * удаляются устаревшие кэши Ultralytics (*.cache).

Запуск: python scripts/fix_labels.py
"""

from pathlib import Path
import shutil

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
IMAGES_ROOT = DATASET_ROOT / "images"
LABELS_ROOT = DATASET_ROOT / "labels"
BACKUP_ROOT = DATASET_ROOT / "labels_dota_backup"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_image(split: str, stem: str) -> Path:
    """Находит изображение, соответствующее TXT-файлу."""
    for extension in IMAGE_EXTENSIONS:
        image_path = IMAGES_ROOT / split / f"{stem}{extension}"

        if image_path.exists():
            return image_path

    raise FileNotFoundError(
        f"Не найдено изображение для labels/{split}/{stem}.txt"
    )


def validate_yolo_line(parts: list[str], file_path: Path, line_number: int) -> None:
    """Проверяет строку стандартного YOLO Detect."""
    if len(parts) != 5:
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            f"ожидалось 5 значений, найдено {len(parts)}"
        )

    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError as error:
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            "не удалось прочитать YOLO-значения"
        ) from error

    if class_id != 0:
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            f"неизвестный класс {class_id}"
        )

    if not 0.0 <= x_center <= 1.0:
        raise ValueError(
            f"{file_path}, строка {line_number}: x_center вне диапазона 0–1"
        )

    if not 0.0 <= y_center <= 1.0:
        raise ValueError(
            f"{file_path}, строка {line_number}: y_center вне диапазона 0–1"
        )

    if not 0.0 < width <= 1.0:
        raise ValueError(
            f"{file_path}, строка {line_number}: width вне диапазона (0–1]"
        )

    if not 0.0 < height <= 1.0:
        raise ValueError(
            f"{file_path}, строка {line_number}: height вне диапазона (0–1]"
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
            f"{file_path}, строка {line_number}: "
            "ошибка чтения координат"
        ) from error

    class_name = parts[8]

    if class_name != "Vehicle":
        raise ValueError(
            f"{file_path}, строка {line_number}: "
            f"неизвестный класс {class_name!r}"
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
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height

    return (
        f"0 "
        f"{x_center:.10f} "
        f"{y_center:.10f} "
        f"{normalized_width:.10f} "
        f"{normalized_height:.10f}"
    )


def determine_format(lines: list[str], file_path: Path) -> str:
    """Определяет формат всего файла: YOLO или DOTA."""
    formats = set()

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) == 5:
            try:
                validate_yolo_line(parts, file_path, line_number)
            except ValueError:
                raise

            formats.add("yolo")

        elif len(parts) == 10 and parts[8] == "Vehicle":
            formats.add("dota")

        else:
            raise ValueError(
                f"{file_path}, строка {line_number}: "
                f"неизвестный формат: {line}"
            )

    if len(formats) > 1:
        raise ValueError(
            f"{file_path}: в одном файле смешаны YOLO и DOTA"
        )

    if not formats:
        return "empty"

    return formats.pop()


def process_split(split: str) -> tuple[int, int, int]:
    """Проверяет и при необходимости конвертирует один split."""
    labels_dir = LABELS_ROOT / split

    if not labels_dir.exists():
        raise FileNotFoundError(f"Не найдена папка: {labels_dir}")

    converted_files = 0
    skipped_yolo_files = 0
    object_count = 0

    label_files = sorted(labels_dir.glob("*.txt"))

    print()
    print(f"Обработка: {split}")
    print(f"TXT-файлов: {len(label_files)}")

    for label_path in label_files:
        raw_lines = label_path.read_text(
            encoding="utf-8",
            errors="strict",
        ).splitlines()

        lines = [line.strip() for line in raw_lines if line.strip()]
        label_format = determine_format(lines, label_path)

        if label_format == "empty":
            print(f"ПРЕДУПРЕЖДЕНИЕ: пустой файл {label_path.name}")
            continue

        if label_format == "yolo":
            skipped_yolo_files += 1
            object_count += len(lines)
            continue

        image_path = find_image(split, label_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise RuntimeError(f"Не удалось открыть изображение: {image_path}")

        image_height, image_width = image.shape[:2]

        converted_lines = []

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            converted_line = convert_dota_line(
                parts=parts,
                image_width=image_width,
                image_height=image_height,
                file_path=label_path,
                line_number=line_number,
            )

            converted_lines.append(converted_line)

        # Сохраняем исходную DOTA-разметку в резервную папку.
        backup_dir = BACKUP_ROOT / split
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_path = backup_dir / label_path.name

        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)

        # Запись через временный файл защищает от частичной перезаписи.
        temporary_path = label_path.with_suffix(".tmp")

        temporary_path.write_text(
            "\n".join(converted_lines) + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(label_path)

        converted_files += 1
        object_count += len(converted_lines)

    return converted_files, skipped_yolo_files, object_count


def validate_all_labels() -> None:
    """Финальная проверка: после обработки разрешён только YOLO Detect."""
    print()
    print("Финальная проверка всех аннотаций...")

    for split in SPLITS:
        labels_dir = LABELS_ROOT / split
        image_dir = IMAGES_ROOT / split

        label_files = sorted(labels_dir.glob("*.txt"))
        image_files = [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(label_files) != len(image_files):
            raise RuntimeError(
                f"{split}: изображений {len(image_files)}, "
                f"аннотаций {len(label_files)}"
            )

        object_count = 0

        for label_path in label_files:
            lines = [
                line.strip()
                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                validate_yolo_line(parts, label_path, line_number)
                object_count += 1

        print(
            f"{split}: "
            f"{len(image_files)} изображений, "
            f"{len(label_files)} аннотаций, "
            f"{object_count} объектов — OK"
        )


def delete_ultralytics_caches() -> None:
    """Удаляет старые кэши, чтобы YOLO пересканировал исправленные TXT."""
    for cache_path in LABELS_ROOT.glob("*.cache"):
        cache_path.unlink()
        print(f"Удалён старый кэш: {cache_path}")


def main() -> None:
    print("Конвертация DOTA/OBB → YOLO Detect HBB")
    print(f"Корень датасета: {DATASET_ROOT}")

    if not DATASET_ROOT.is_dir():
        raise SystemExit(
            f"Не найден подготовленный датасет: {DATASET_ROOT}\n"
            "Сначала выполните: python scripts/prepare_dataset.py --source <путь к SIVED>"
        )

    for split in SPLITS:
        converted, skipped, objects = process_split(split)

        print(
            f"{split}: "
            f"сконвертировано файлов — {converted}, "
            f"уже были YOLO — {skipped}, "
            f"объектов — {objects}"
        )

    validate_all_labels()
    delete_ultralytics_caches()

    print()
    print("Готово. Все аннотации имеют формат YOLO Detect.")


if __name__ == "__main__":
    main()