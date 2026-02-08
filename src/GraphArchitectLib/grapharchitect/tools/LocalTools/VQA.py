import torch
import logging
from typing import Union, Optional
from transformers import BlipForConditionalGeneration, BlipProcessor, PreTrainedModel
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerativeVQA:
    """
    Класс для выполнения визуального вопрос-ответ (VQA) с использованием генеративной модели.
    Здесь используется модель BLIP, способная генерировать ответ на вопрос по изображению.
    Можно передать либо название/путь до модели, либо уже загруженную модель с соответствующим процессором.
    """

    def __init__(
        self,
        model_or_path: Union[str, PreTrainedModel],
        processor: Optional[BlipProcessor] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Инициализация генеративной модели VQA.

        Args:
            model_or_path (Union[str, PreTrainedModel]): Название или путь до модели,
                либо уже загруженная модель.
            processor (Optional[BlipProcessor], optional): Процессор, объединяющий обработку изображений
                и текста. Обязателен, если передана модель, а не строка.
            device (Optional[str], optional): Устройство для вычислений ('cuda' или 'cpu').
                Если не указано, выбирается автоматически.
        """
        if isinstance(model_or_path, str):
            self.model = BlipForConditionalGeneration.from_pretrained(model_or_path)
            self.processor = BlipProcessor.from_pretrained(model_or_path)
        else:
            self.model = model_or_path
            if processor is None:
                raise ValueError("При передаче модели необходимо указать и процессор.")
            self.processor = processor

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def answer(self, image: Image.Image, question: str, max_length: int = 50, **generate_kwargs) -> str:
        """
        Генерирует ответ на заданный вопрос по изображению.

        Args:
            image (PIL.Image.Image): Изображение для анализа.
            question (str): Вопрос, связанный с изображением.
            max_length (int, optional): Максимальная длина генерируемого ответа.
            **generate_kwargs: Дополнительные параметры для метода generate модели.

        Returns:
            str: Сгенерированный ответ на вопрос.
        """
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length, **generate_kwargs)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return answer
