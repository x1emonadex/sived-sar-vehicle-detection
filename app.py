from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import streamlit as st
import torch
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "sived_yolo12n_best.pt"

TARGET_WIDTH = 1024
YOLO_IMAGE_SIZE = 512
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 МБ
REQUEST_TIMEOUT = (5, 30)


# ---------------------------------------------------------
# Загрузка и декодирование
# ---------------------------------------------------------
def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Получен пустой файл.")

    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise ValueError("OpenCV не смог прочитать изображение.")

    return image


def load_image_from_url(url: str) -> tuple[bytes, str]:
    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Введите корректную ссылку, начинающуюся с http:// или https://."
        )

    try:
        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as response:
            response.raise_for_status()

            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > MAX_FILE_SIZE:
                raise ValueError("Файл по URL превышает допустимые 100 МБ.")

            chunks: list[bytes] = []
            total_size = 0

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise ValueError("Файл по URL превышает допустимые 100 МБ.")

                chunks.append(chunk)

    except requests.RequestException as error:
        raise ValueError(f"Не удалось загрузить изображение: {error}") from error

    data = b"".join(chunks)
    filename = parsed.path.rsplit("/", 1)[-1] or "image_from_url"

    return data, filename


# ---------------------------------------------------------
# Обработка изображения
# ---------------------------------------------------------
def to_display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    channels = image.shape[2]

    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    raise ValueError("Неподдерживаемое количество цветовых каналов.")


def resize_to_1024(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    if width <= TARGET_WIDTH:
        return image.copy()

    new_height = max(1, round(height * TARGET_WIDTH / width))

    return cv2.resize(
        image,
        (TARGET_WIDTH, new_height),
        interpolation=cv2.INTER_AREA,
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()

    channels = image.shape[2]

    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError("Неподдерживаемое количество цветовых каналов.")


def apply_threshold(
    gray: np.ndarray,
    method: str,
    threshold_value: int,
    block_size: int,
    c_value: int,
) -> tuple[np.ndarray, str]:
    if method == "Бинарный":
        _, result = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY,
        )
        description = f"Бинарный порог: {threshold_value}"

    elif method == "Обратный бинарный":
        _, result = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )
        description = f"Обратный бинарный порог: {threshold_value}"

    elif method == "Оцу":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        otsu_value, result = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        description = f"Автоматический порог Оцу: {otsu_value:.0f}"

    elif method == "Адаптивный средний":
        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )
        description = (
            f"Адаптивный средний · блок {block_size} · C = {c_value}"
        )

    elif method == "Адаптивный гауссов":
        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )
        description = (
            f"Адаптивный гауссов · блок {block_size} · C = {c_value}"
        )

    else:
        raise ValueError("Неизвестный метод порогового преобразования.")

    return result, description


def apply_canny(
    gray: np.ndarray,
    lower_threshold: int,
    upper_threshold: int,
    blur_kernel: int,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        gray,
        (blur_kernel, blur_kernel),
        0,
    )

    return cv2.Canny(
        blurred,
        lower_threshold,
        upper_threshold,
    )


def make_preview(
    image: np.ndarray,
    max_width: int = 1120,
    max_height: int = 500,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return image

    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))

    return cv2.resize(
        image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )


def make_small_preview(
    image: np.ndarray,
    max_width: int = 500,
    max_height: int = 240,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )

    if not success:
        raise ValueError("Не удалось подготовить PNG-файл для скачивания.")

    return encoded.tobytes()


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size_bytes} Б"


def make_image_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------
# YOLO12n: загрузка модели и обнаружение
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str) -> YOLO:
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Файл модели не найден: {path}"
        )

    return YOLO(str(path))


def get_inference_device() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def get_device_name(device: int | str) -> str:
    if device == "cpu":
        return "CPU"

    return torch.cuda.get_device_name(0)


def detect_vehicles(
    model: YOLO,
    gray: np.ndarray,
    confidence: float,
    iou_threshold: float,
) -> dict[str, object]:
    # Модель обучалась на РЛИ-изображениях в оттенках серого.
    # Ultralytics принимает трёхканальное изображение, поэтому один
    # серый канал повторяется в BGR без изменения яркостной информации.
    model_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    device = get_inference_device()

    started_at = time.perf_counter()

    results = model.predict(
        source=model_input,
        imgsz=YOLO_IMAGE_SIZE,
        conf=confidence,
        iou=iou_threshold,
        max_det=300,
        device=device,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    result = results[0]

    annotated_bgr = result.plot(
        labels=False,
        conf=False,
        line_width=2,
    )
    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

    detections: list[dict[str, object]] = []

    if result.boxes is not None and len(result.boxes) > 0:
        coordinates = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        for index, (xyxy, score, class_id) in enumerate(
            zip(coordinates, confidences, class_ids),
            start=1,
        ):
            x1, y1, x2, y2 = xyxy.tolist()

            if isinstance(result.names, dict):
                class_name = result.names.get(class_id, str(class_id))
            else:
                class_name = result.names[class_id]

            detections.append(
                {
                    "№": index,
                    "Класс": str(class_name),
                    "Уверенность": float(score),
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                }
            )

    return {
        "annotated_rgb": annotated_rgb,
        "detections": detections,
        "device_name": get_device_name(device),
        "elapsed_ms": elapsed_ms,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
    }


# ---------------------------------------------------------
# Оформление
# ---------------------------------------------------------
def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 1.5rem;
        }

        h1 {
            letter-spacing: -0.03em;
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.12);
        }

        div[data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
        }

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .sidebar-note {
            color: #667085;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-bottom: 0.8rem;
        }

        .comparison-title {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }

        .model-path {
            color: #667085;
            font-size: 0.76rem;
            line-height: 1.35;
            word-break: break-all;
            margin-top: -0.25rem;
        }

        div[data-testid="stImage"] img {
            border-radius: 0.65rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }

        section[data-testid="stFileUploaderDropzone"] {
            border-radius: 0.75rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 0.75rem;
        }

        button[data-baseweb="tab"] {
            font-size: 0.9rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, int | float | str | bool]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">Параметры обработки</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-note">'
            'Здесь выбираются параметры классической обработки '
            'и обнаружения транспорта.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Пороговое преобразование")

        threshold_method = st.selectbox(
            "Метод",
            [
                "Бинарный",
                "Обратный бинарный",
                "Оцу",
                "Адаптивный средний",
                "Адаптивный гауссов",
            ],
        )

        threshold_value = 127
        block_size = 11
        c_value = 2

        if threshold_method in {"Бинарный", "Обратный бинарный"}:
            threshold_value = st.slider(
                "Порог яркости",
                min_value=0,
                max_value=255,
                value=127,
            )

        elif threshold_method in {
            "Адаптивный средний",
            "Адаптивный гауссов",
        }:
            block_size = st.slider(
                "Размер блока",
                min_value=3,
                max_value=51,
                value=11,
                step=2,
                help="Должен быть нечётным числом.",
            )
            c_value = st.slider(
                "Поправка C",
                min_value=-100,
                max_value=100,
                value=2,
            )

        st.divider()
        st.subheader("Метод Канни")

        canny_lower, canny_upper = st.slider(
            "Пороги Канни",
            min_value=0,
            max_value=255,
            value=(50, 150),
            help="Первое значение — нижний порог, второе — верхний.",
        )

        blur_kernel = st.select_slider(
            "Сглаживание",
            options=[3, 5, 7, 9, 11, 13, 15],
            value=9,
            help="Размер ядра гауссова фильтра перед методом Канни.",
        )

        st.divider()
        st.subheader("Обнаружение YOLO12n")

        if MODEL_PATH.is_file():
            st.success("Модель найдена и готова.")
        else:
            st.error("Файл модели не найден.")

        st.markdown(
            f'<div class="model-path">{MODEL_PATH}</div>',
            unsafe_allow_html=True,
        )

        confidence = st.slider(
            "Порог уверенности",
            min_value=0.05,
            max_value=0.95,
            value=0.27,
            step=0.01,
            help=(
                "0.27 — рекомендуемая начальная точка "
                "по F1-кривой тестовой выборки."
            ),
        )

        iou_threshold = st.slider(
            "IoU для удаления дубликатов",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help=(
                "Меньшее значение сильнее подавляет "
                "перекрывающиеся рамки."
            ),
        )

        run_detection_sidebar = st.button(
            "Запустить нейросеть",
            type="primary",
            use_container_width=True,
            disabled=not MODEL_PATH.is_file(),
            key="run_yolo_sidebar",
        )

        st.caption(
            "Нейросеть запускается только вручную. "
            "Изменение параметров Канни и пороговой обработки "
            "не вызывает повторный запуск YOLO."
        )

    return {
        "threshold_method": threshold_method,
        "threshold_value": threshold_value,
        "block_size": block_size,
        "c_value": c_value,
        "canny_lower": canny_lower,
        "canny_upper": canny_upper,
        "blur_kernel": blur_kernel,
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "run_detection_sidebar": run_detection_sidebar,
    }


# ---------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Обработка РЛИ-изображений",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_styles()
    settings = render_sidebar()

    st.title("Обработка и анализ РЛИ-изображений")
    st.write(
        "Загрузите изображение с компьютера или по URL. "
        "Приложение уменьшит ширину до 1024 пикселей, переведёт изображение "
        "в оттенки серого, выполнит пороговую обработку, выделит границы "
        "методом Канни и сможет обнаружить транспорт моделью YOLO12n."
    )

    source_type = st.radio(
        "Источник изображения",
        ["Файл с компьютера", "URL-ссылка"],
        horizontal=True,
    )

    image_data: bytes | None = None
    filename = ""

    try:
        if source_type == "Файл с компьютера":
            uploaded_file = st.file_uploader(
                "Выберите РЛИ-изображение",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
            )

            if uploaded_file is not None:
                image_data = uploaded_file.getvalue()
                filename = uploaded_file.name

        else:
            image_url = st.text_input(
                "Введите прямую ссылку на изображение",
                placeholder="https://example.com/image.jpg",
            )

            if st.button(
                "Загрузить изображение по URL",
                type="primary",
                use_container_width=True,
            ):
                if not image_url.strip():
                    st.warning("Сначала введите URL изображения.")
                else:
                    with st.spinner("Загрузка изображения..."):
                        data, url_filename = load_image_from_url(image_url)
                        st.session_state["url_image_data"] = data
                        st.session_state["url_image_name"] = url_filename

            image_data = st.session_state.get("url_image_data")
            filename = st.session_state.get("url_image_name", "")

        if image_data is None:
            st.info(
                "Выберите изображение с компьютера или загрузите его по URL."
            )
            return

        if len(image_data) > MAX_FILE_SIZE:
            raise ValueError("Размер файла превышает допустимые 100 МБ.")

        image_id = make_image_id(image_data)
        original = decode_image(image_data)
        original_height, original_width = original.shape[:2]

        resized_color = resize_to_1024(original)
        gray = to_grayscale(resized_color)

        threshold_image, threshold_description = apply_threshold(
            gray=gray,
            method=str(settings["threshold_method"]),
            threshold_value=int(settings["threshold_value"]),
            block_size=int(settings["block_size"]),
            c_value=int(settings["c_value"]),
        )

        canny_image = apply_canny(
            gray=gray,
            lower_threshold=int(settings["canny_lower"]),
            upper_threshold=int(settings["canny_upper"]),
            blur_kernel=int(settings["blur_kernel"]),
        )

        processed_height, processed_width = gray.shape[:2]

        saved_yolo_result = st.session_state.get("yolo_result")
        yolo_result: dict[str, object] | None = None

        if (
            isinstance(saved_yolo_result, dict)
            and saved_yolo_result.get("image_id") == image_id
        ):
            yolo_result = saved_yolo_result

        if bool(settings["run_detection_sidebar"]):
            try:
                with st.spinner("Модель анализирует изображение..."):
                    model = load_yolo_model(str(MODEL_PATH))
                    new_result = detect_vehicles(
                        model=model,
                        gray=gray,
                        confidence=float(settings["confidence"]),
                        iou_threshold=float(settings["iou_threshold"]),
                    )
                    new_result["image_id"] = image_id
                    st.session_state["yolo_result"] = new_result
                    yolo_result = new_result

            except Exception as error:
                st.error(
                    f"Не удалось выполнить обнаружение: {error}"
                )

        st.success("Изображение успешно загружено и обработано.")

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Источник", filename)
        metric_2.metric(
            "Исходный размер",
            f"{original_width} × {original_height}",
        )
        metric_3.metric(
            "Размер обработки",
            f"{processed_width} × {processed_height}",
        )
        metric_4.metric(
            "Обнаружено объектов",
            (
                len(yolo_result["detections"])
                if yolo_result is not None
                else "—"
            ),
        )

        tabs = st.tabs(
            [
                "Исходное",
                "Оттенки серого",
                "Пороговое",
                "Канни",
                "Обнаружение YOLO",
                "Сравнение обработки",
                "Сравнение YOLO",
            ]
        )

        with tabs[0]:
            st.image(
                make_preview(to_display_image(original)),
                caption=(
                    f"Исходное изображение — "
                    f"{original_width} × {original_height}"
                ),
                use_container_width=True,
            )

        with tabs[1]:
            st.image(
                make_preview(gray),
                caption=(
                    f"Оттенки серого — "
                    f"{processed_width} × {processed_height}"
                ),
                use_container_width=True,
                clamp=True,
            )

        with tabs[2]:
            st.image(
                make_preview(threshold_image),
                caption=threshold_description,
                use_container_width=True,
                clamp=True,
            )

        with tabs[3]:
            st.image(
                make_preview(canny_image),
                caption=(
                    f"Метод Канни — пороги "
                    f"{settings['canny_lower']} и {settings['canny_upper']}"
                ),
                use_container_width=True,
                clamp=True,
            )

        with tabs[4]:
            st.subheader("Обнаружение транспортных средств")
            st.caption(
                "Модель запускается вручную. Изменение параметров Канни "
                "или пороговой обработки не вызывает повторный запуск YOLO."
            )

            run_detection = st.button(
                "Запустить обнаружение",
                type="primary",
                use_container_width=True,
                disabled=not MODEL_PATH.is_file(),
                key="run_yolo_tab",
            )

            if run_detection:
                try:
                    with st.spinner(
                        "Модель анализирует изображение..."
                    ):
                        model = load_yolo_model(str(MODEL_PATH))
                        new_result = detect_vehicles(
                            model=model,
                            gray=gray,
                            confidence=float(settings["confidence"]),
                            iou_threshold=float(settings["iou_threshold"]),
                        )
                        new_result["image_id"] = image_id
                        st.session_state["yolo_result"] = new_result
                        yolo_result = new_result
                        st.rerun()

                except Exception as error:
                    st.error(
                        f"Не удалось выполнить обнаружение: {error}"
                    )

            if yolo_result is None:
                st.info(
                    "Для этого изображения обнаружение ещё не запускалось."
                )

            else:
                detections = yolo_result["detections"]
                annotated_rgb = yolo_result["annotated_rgb"]
                confidences = [
                    float(item["Уверенность"])
                    for item in detections
                ]

                yolo_metric_1, yolo_metric_2, yolo_metric_3, yolo_metric_4 = (
                    st.columns(4)
                )
                yolo_metric_1.metric("Объекты", len(detections))
                yolo_metric_2.metric(
                    "Средняя уверенность",
                    (
                        f"{np.mean(confidences):.3f}"
                        if confidences
                        else "—"
                    ),
                )
                yolo_metric_3.metric(
                    "Устройство",
                    str(yolo_result["device_name"]),
                )
                yolo_metric_4.metric(
                    "Время запуска",
                    f"{float(yolo_result['elapsed_ms']):.1f} мс",
                )

                st.image(
                    make_preview(annotated_rgb),
                    caption=(
                        "Результат YOLO12n · "
                        f"confidence ≥ "
                        f"{float(yolo_result['confidence']):.2f} · "
                        f"IoU = "
                        f"{float(yolo_result['iou_threshold']):.2f}"
                    ),
                    use_container_width=True,
                )

                if detections:
                    with st.expander(
                        f"Таблица обнаружений ({len(detections)})",
                        expanded=False,
                    ):
                        st.dataframe(
                            detections,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Уверенность": st.column_config.ProgressColumn(
                                    "Уверенность",
                                    min_value=0.0,
                                    max_value=1.0,
                                    format="%.3f",
                                ),
                            },
                        )
                else:
                    st.warning(
                        "При выбранном пороге уверенности "
                        "транспорт не обнаружен."
                    )

        with tabs[5]:
            st.subheader("Сравнение классической обработки")

            row_1_left, row_1_right = st.columns(2)

            with row_1_left:
                st.markdown(
                    '<div class="comparison-title">'
                    'Цветное / исходное'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    make_small_preview(
                        to_display_image(resized_color)
                    ),
                    use_container_width=True,
                )

            with row_1_right:
                st.markdown(
                    '<div class="comparison-title">'
                    'Оттенки серого'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    make_small_preview(gray),
                    use_container_width=True,
                    clamp=True,
                )

            row_2_left, row_2_right = st.columns(2)

            with row_2_left:
                st.markdown(
                    '<div class="comparison-title">'
                    'Пороговое преобразование'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    make_small_preview(threshold_image),
                    use_container_width=True,
                    clamp=True,
                )

            with row_2_right:
                st.markdown(
                    '<div class="comparison-title">'
                    'Метод Канни'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    make_small_preview(canny_image),
                    use_container_width=True,
                    clamp=True,
                )

        with tabs[6]:
            st.subheader("Сравнение с результатом нейросети")

            compare_original, compare_gray, compare_yolo = st.columns(3)

            with compare_original:
                st.markdown(
                    '<div class="comparison-title">'
                    'Цветное / исходное'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    make_small_preview(
                        to_display_image(resized_color)
                    ),
                    use_container_width=True,
                )

            with compare_gray:
                st.markdown(
                    '<div class="comparison-title">'
                    'Оттенки серого'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    make_small_preview(gray),
                    use_container_width=True,
                    clamp=True,
                )

            with compare_yolo:
                st.markdown(
                    '<div class="comparison-title">'
                    'Результат YOLO12n'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if yolo_result is not None:
                    st.image(
                        make_small_preview(
                            yolo_result["annotated_rgb"]
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Запустите нейросеть кнопкой слева "
                        "или во вкладке «Обнаружение YOLO»."
                    )

        status_text = (
            f"Пороговая обработка: {threshold_description}. "
            f"Канни: нижний порог {settings['canny_lower']}, "
            f"верхний порог {settings['canny_upper']}."
        )

        if yolo_result is not None:
            status_text += (
                f" YOLO12n: найдено "
                f"{len(yolo_result['detections'])} объектов."
            )

        st.info(status_text)

        base_name = filename.rsplit(".", 1)[0] or "image"

        if yolo_result is None:
            download_1, download_2, download_3 = st.columns(3)
            download_4 = None
        else:
            download_1, download_2, download_3, download_4 = st.columns(4)

        with download_1:
            st.download_button(
                "Скачать серое",
                data=encode_png(gray),
                file_name=f"{base_name}_gray.png",
                mime="image/png",
                use_container_width=True,
            )

        with download_2:
            st.download_button(
                "Скачать пороговое",
                data=encode_png(threshold_image),
                file_name=f"{base_name}_threshold.png",
                mime="image/png",
                use_container_width=True,
            )

        with download_3:
            st.download_button(
                "Скачать Канни",
                data=encode_png(canny_image),
                file_name=f"{base_name}_canny.png",
                mime="image/png",
                use_container_width=True,
            )

        if download_4 is not None and yolo_result is not None:
            with download_4:
                detection_bgr = cv2.cvtColor(
                    yolo_result["annotated_rgb"],
                    cv2.COLOR_RGB2BGR,
                )

                st.download_button(
                    "Скачать YOLO",
                    data=encode_png(detection_bgr),
                    file_name=f"{base_name}_yolo.png",
                    mime="image/png",
                    use_container_width=True,
                )

    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Непредвиденная ошибка: {error}")


if __name__ == "__main__":
    main()
