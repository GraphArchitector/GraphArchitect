import requests
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InfinityEmbedder:
    """
    Клиент для получения эмбеддингов от эмбеддера, развернутого на Infinity.
    """

    def __init__(self, base_url: str, api_key: str = None) -> None:
        """
        Инициализация клиента InfinityEmbedder.

        Args:
            base_url (str): Базовый URL для доступа к API эмбеддера.
            api_key (str, optional): API ключ, если требуется.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.endpoint = f"{self.base_url}/embeddings"  # Пример эндпоинта
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def get_embedding(self, text: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Получает эмбеддинг для заданного текста.

        Args:
            text (str): Текст, для которого нужно получить эмбеддинг.
            timeout (int, optional): Таймаут запроса в секундах. По умолчанию 10.

        Returns:
            Dict[str, Any]: Словарь с данными эмбеддинга или сообщение об ошибке.
        """
        payload: Dict[str, Any] = {
            "input": text,
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка при получении эмбеддинга: %s", e)
            return {"error": str(e)}
        except ValueError as e:
            logger.error("Ошибка при разборе ответа: %s", e)
            return {"error": str(e)}