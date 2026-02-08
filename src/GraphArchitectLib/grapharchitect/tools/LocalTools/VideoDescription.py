import torch
import logging
from typing import Union, Optional, List
from transformers import AutoModelForConditionalGeneration, PreTrainedModel
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VideoDescriber:
    """
    Класс для генерации описания видео с использованием генеративной модели.
    Можно передать либо название/путь до модели, либо уже загруженную модель с соответствующим процессором.
    Обратите внимание, что для корректной работы требуется процессор, способный обрабатывать последовательности кадров.
    """

    def __init__(
        self,
        model_or_path: Union[str, PreTrainedModel],
        processor: Optional[object] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Инициализация модели для описания видео.

        Args:
            model_or_path (Union[str, PreTrainedModel]): Название или путь до модели,
                либо уже загруженная модель.
            processor (Optional[object], optional): Процессор для обработки видео (например, для извлечения кадров и подготовки входных данных).
                Обязателен, если передана модель, а не строка.
            device (Optional[str], optional): Устройство для вычислений ('cuda' или 'cpu').
                Если не указано, выбирается автоматически.
        """
        if isinstance(model_or_path, str):
            self.model = AutoModelForConditionalGeneration.from_pretrained(model_or_path)
            if processor is None:
                raise ValueError("При загрузке модели по имени необходимо передать и процессор.")
            self.processor = processor
        else:
            self.model = model_or_path
            if processor is None:
                raise ValueError("При передаче модели необходимо указать и процессор.")
            self.processor = processor

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def _extract_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """
        Извлекает ключевые кадры из видео.

        Args:
            video_path (str): Путь к видеофайлу.
            num_frames (int, optional): Количество извлекаемых кадров. По умолчанию 8.

        Returns:
            List[Image.Image]: Список объектов PIL.Image с извлечёнными кадрами.
        """
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Не удалось определить количество кадров в видео.")
        # Равномерно выбираем num_frames кадров
        frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                # Конвертируем кадр из BGR (OpenCV) в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                frames.append(image)
        cap.release()
        return frames

    def describe(self, video: Union[str, List[Image.Image]], max_length: int = 50, **generate_kwargs) -> str:
        """
        Генерирует описание для заданного видео.

        Args:
            video (Union[str, List[Image.Image]]): Путь к видеофайлу или список кадров (объекты PIL.Image).
            max_length (int, optional): Максимальная длина генерируемого описания.
            **generate_kwargs: Дополнительные параметры для метода generate модели.

        Returns:
            str: Сгенерированное описание видео.
        """
        # Если передан путь к видео, извлекаем кадры
        if isinstance(video, str):
            frames = self._extract_frames(video)
        elif isinstance(video, list):
            frames = video
        else:
            raise ValueError("Видео должно быть либо путем к файлу, либо списком кадров (PIL.Image).")

        # Предполагается, что процессор может обрабатывать список кадров
        inputs = self.processor(frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_length=max_length, **generate_kwargs)
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        return description
