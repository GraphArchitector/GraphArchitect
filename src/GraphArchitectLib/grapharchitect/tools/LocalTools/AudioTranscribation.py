import torch
import logging
from typing import Union, Optional
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, PreTrainedModel
import librosa
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AudioRecognizer:
    """
    Класс для распознавания речи с использованием модели Wav2Vec2.
    Можно передать либо название/путь до модели, либо уже загруженную модель с соответствующим процессором.
    """

    def __init__(
            self,
            model_or_path: Union[str, PreTrainedModel],
            processor: Optional[Wav2Vec2Processor] = None,
            device: Optional[str] = None
    ) -> None:
        """
        Инициализация аудио-распознавателя.

        Args:
            model_or_path (Union[str, PreTrainedModel]): Название или путь до модели,
                либо уже загруженная модель.
            processor (Optional[Wav2Vec2Processor], optional): Процессор для обработки аудио.
                Обязателен, если передана модель, а не строка.
            device (Optional[str], optional): Устройство для вычислений ('cuda' или 'cpu').
                Если не указано, выбирается автоматически.
        """
        if isinstance(model_or_path, str):
            self.model = Wav2Vec2ForCTC.from_pretrained(model_or_path)
            self.processor = Wav2Vec2Processor.from_pretrained(model_or_path)
        else:
            self.model = model_or_path
            if processor is None:
                raise ValueError("При передаче модели необходимо указать и процессор.")
            self.processor = processor

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def transcribe(self, audio: Union[str, np.ndarray], sampling_rate: int = 16000) -> str:
        """
        Выполняет распознавание речи на основе аудио.

        Args:
            audio (Union[str, np.ndarray]): Аудио данные в виде пути к аудиофайлу или numpy массива.
            sampling_rate (int, optional): Частота дискретизации аудио. По умолчанию 16000 Гц.

        Returns:
            str: Распознанная текстовая транскрипция.
        """
        if isinstance(audio, str):
            # Загружаем аудио из файла с нужной частотой дискретизации
            audio_array, _ = librosa.load(audio, sr=sampling_rate)
        else:
            audio_array = audio

        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]
