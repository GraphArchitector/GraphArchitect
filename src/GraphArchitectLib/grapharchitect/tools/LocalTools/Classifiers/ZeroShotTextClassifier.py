import torch
import logging
from typing import Union, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ZeroShotClassifier:
    """
    Класс для выполнения zero-shot классификации с использованием модели MNLI.
    Можно передать либо название/путь до модели, либо уже загруженную модель с токенайзером.
    """

    def __init__(
        self,
        model_or_path: Union[str, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Инициализация zero-shot классификатора.

        Args:
            model_or_path (Union[str, PreTrainedModel]): Название или путь до модели,
                либо уже загруженная модель.
            tokenizer (Optional[PreTrainedTokenizer], optional): Токенайзер для модели.
                Обязателен, если передана модель, а не строка.
            device (Optional[str], optional): Устройство для вычислений ('cuda' или 'cpu').
                Если не указано, выбирается автоматически.
        """
        if isinstance(model_or_path, str):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)
        else:
            self.model = model_or_path
            if tokenizer is None:
                raise ValueError("При передаче модели необходимо указать и токенайзер.")
            self.tokenizer = tokenizer

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def classify(self, sequence: str, label: str) -> float:
        """
        Выполняет zero-shot классификацию текста и возвращает вероятность того,
        что текст соответствует заданной метке.

        Args:
            sequence (str): Текст для классификации.
            label (str): Гипотетическая метка (например, "positive" или "negative").

        Returns:
            float: Вероятность того, что текст соответствует метке.
        """
        premise = sequence
        hypothesis = f"This example is {label}."
        inputs = self.tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation=True).to(self.device)
        logits = self.model(inputs)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        return probs[:, 1].item()