import torch
import logging
from typing import Union, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextGenerator:
    """
    Класс для генерации текста с использованием предобученной генеративной модели.
    Можно передать либо название/путь до модели, либо уже загруженную модель с токенайзером.
    """

    def __init__(
        self,
        model_or_path: Union[str, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Инициализация генератора текста.

        Args:
            model_or_path (Union[str, PreTrainedModel]): Название или путь до модели,
                либо уже загруженная модель.
            tokenizer (Optional[PreTrainedTokenizer], optional): Токенайзер для модели.
                Обязателен, если передана модель, а не строка.
            device (Optional[str], optional): Устройство для вычислений ('cuda' или 'cpu').
                Если не указано, выбирается автоматически.
        """
        if isinstance(model_or_path, str):
            self.model = AutoModelForCausalLM.from_pretrained(model_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)
        else:
            self.model = model_or_path
            if tokenizer is None:
                raise ValueError("При передаче модели необходимо указать и токенайзер.")
            self.tokenizer = tokenizer

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        **generate_kwargs
    ) -> Union[str, List[str]]:
        """
        Генерирует текст на основе заданного пролога.

        Args:
            prompt (str): Начальный текст для генерации.
            max_length (int, optional): Максимальная длина генерируемой последовательности.
            num_return_sequences (int, optional): Количество вариантов генерации.
            **generate_kwargs: Дополнительные параметры для метода generate модели.

        Returns:
            Union[str, List[str]]: Сгенерированный текст, если num_return_sequences == 1,
                                   иначе список сгенерированных текстов.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **generate_kwargs
        )
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts[0] if num_return_sequences == 1 else generated_texts
