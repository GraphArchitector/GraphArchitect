import requests
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatGPTApi:
    """
    Клиент для взаимодействия с ChatGPT API посредством прямых HTTP запросов.
    """

    def __init__(self, api_key: str, model_name: str, system_prompt: str) -> None:
        """
        Инициализация клиента ChatGPTApi.

        Args:
            api_key (str): API ключ OpenAI.
            model_name (str): Имя модели (например, 'gpt-3.5-turbo').
            system_prompt (str): Системное сообщение для модели.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def query_llm(
        self,
        question: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 500,
        timeout: int = 10
    ) -> str:
        """
        Отправляет запрос к ChatGPT API и возвращает ответ модели.

        Args:
            question (str): Вопрос или сообщение пользователя.
            temperature (float, optional): Параметр temperature для сэмплирования. По умолчанию 0.2.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.95.
            max_tokens (int, optional): Максимальное число токенов в ответе. По умолчанию 500.
            timeout (int, optional): Таймаут запроса в секундах. По умолчанию 10.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка при выполнении запроса к ChatGPT API: %s", e)
            return f"Request failed: {e}"
        except (KeyError, IndexError) as e:
            logger.error("Неожиданный формат ответа: %s", e)
            return f"Unexpected response format: {e}"
