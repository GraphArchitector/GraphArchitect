import requests
import logging
from typing import Any, Dict

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VLLMApi:
    """
    Клиент для взаимодействия с VLLM API для работы с языковыми моделями.
    """

    def __init__(self, vllm_host: str, model_name: str, system_prompt: str) -> None:
        """
        Инициализация клиента VLLMApi.

        Args:
            vllm_host (str): URL сервера VLLM API.
            model_name (str): Имя модели для запросов.
            prompt (str): Системное сообщение для модели.
        """
        self.vllm_host = vllm_host
        self.model_name = model_name
        self.prompt = system_prompt
        self.headers = {"Content-Type": "application/json"}

    def query_llm(
        self,
        question: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 30,
        max_tokens: int = 5000,
        timeout: int = 10
    ) -> str:
        """
        Отправляет запрос к языковой модели.

        Args:
            question (str): Вопрос или сообщение пользователя.
            temperature (float, optional): Параметр temperature для сэмплирования. По умолчанию 0.2.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.95.
            top_k (int, optional): Параметр top-k sampling. По умолчанию 30.
            max_tokens (int, optional): Максимальное число токенов в ответе. По умолчанию 5000.
            timeout (int, optional): Таймаут запроса в секундах. По умолчанию 10.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        payload: Dict[str, Any] = {
            "repetition_penalty": 1.06,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": question}
            ],
            "model": self.model_name
        }

        try:
            response = requests.post(self.vllm_host, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка при выполнении запроса: %s", e)
            return f"Request failed: {e}"
        except (KeyError, IndexError) as e:
            logger.error("Неожиданный формат ответа: %s", e)
            return f"Unexpected response format: {e}"
