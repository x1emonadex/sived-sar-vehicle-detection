from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_yaml = project_dir / "sived.yaml"
    dataset_dir = project_dir / "dataset"

    if not (dataset_dir / "images" / "train").is_dir():
        raise SystemExit(
            "Не найден подготовленный датасет: "
            f"{dataset_dir / 'images' / 'train'}\n"
            "Сначала выполните: python scripts/prepare_dataset.py --source <путь к SIVED>"
        )

    cuda_available = torch.cuda.is_available()

    print("CUDA available:", cuda_available)

    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))
        device: int | str = 0
    else:
        print("GPU not detected. CPU will be used.")
        device = "cpu"

    model = YOLO("yolo12n.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=100,
        imgsz=512,
        batch=8,
        device=device,
        workers=0,
        plots=True,
        save=True,
        project=str(project_dir / "runs" / "detect"),
        name="sived_yolo12n_100",
    )


if __name__ == "__main__":
    main()
