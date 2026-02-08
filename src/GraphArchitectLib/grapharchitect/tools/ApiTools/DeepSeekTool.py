import requests
import logging
from typing import Any, Dict

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DeepSeekApi:
    """
    Клиент для взаимодействия с DeepSeek API для получения ответов от языковой модели.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        base_url: str = "https://api.deepseek.com/v1"
    ) -> None:
        """
        Инициализация клиента DeepSeekApi.

        Args:
            api_key (str): API ключ DeepSeek.
            model_name (str): Имя модели (например, "deepseek-chat" для обычного чата или "deepseek-reasoner" для модели рассуждений).
            system_prompt (str): Системное сообщение для задания контекста.
            base_url (str, optional): Базовый URL для DeepSeek API. По умолчанию "https://api.deepseek.com/v1".
        """
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.endpoint = f"{self.base_url}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def query_llm(
        self,
        question: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 500,
        timeout: int = 10
    ) -> str:
        """
        Отправляет запрос к DeepSeek API и возвращает ответ модели.

        Args:
            question (str): Вопрос или сообщение пользователя.
            temperature (float, optional): Параметр temperature для генерации. По умолчанию 0.7.
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
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            # Извлекаем ответ из структуры JSON
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка при выполнении запроса к DeepSeek API: %s", e)
            return f"Request failed: {e}"
        except (KeyError, IndexError) as e:
            logger.error("Неожиданный формат ответа: %s", e)
            return f"Unexpected response format: {e}"