import torch
import logging
import numpy as np
from typing import Union, Optional, List
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TextVectorizer:
    """
    Класс для векторизации текстов с использованием модели E5.
    Можно передать либо название/путь до модели, либо уже загруженную модель с токенайзером.
    """

    def __init__(
        self,
        model_or_path: Union[str, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Инициализация векторизатора текста.

        Args:
            model_or_path (Union[str, PreTrainedModel]): Название или путь до модели,
                либо уже загруженная модель.
            tokenizer (Optional[PreTrainedTokenizer], optional): Токенайзер для модели.
                Обязателен, если передана модель, а не строка.
            device (Optional[str], optional): Устройство для вычислений ('cuda' или 'cpu').
                Если не указано, выбирается автоматически.
        """
        if isinstance(model_or_path, str):
            self.model = AutoModel.from_pretrained(model_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)
        else:
            self.model = model_or_path
            if tokenizer is None:
                raise ValueError("При передаче модели необходимо указать и токенайзер.")
            self.tokenizer = tokenizer

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """
        Выполняет усреднение эмбеддингов токенов с учётом маски внимания.

        Args:
            model_output: Выход модели с атрибутом last_hidden_state.
            attention_mask: Маска внимания, указывающая реальные токены.

        Returns:
            torch.Tensor: Усреднённые эмбеддинги для каждого примера.
        """
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    def vectorize(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Векторизует один или несколько текстов, возвращая их эмбеддинги.

        Args:
            texts (Union[str, List[str]]): Текст или список текстов для векторизации.

        Returns:
            np.ndarray: Если передан один текст, возвращается 1D-вектор эмбеддинга,
                        если список текстов – матрица эмбеддингов (каждая строка соответствует тексту).
        """
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        embeddings = embeddings.detach().cpu().numpy()
        if len(texts) == 1:
            return embeddings[0]
        return embeddings
