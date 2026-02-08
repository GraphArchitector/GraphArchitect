import requests
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LLMReplicateApi:
    """
    Клиент для взаимодействия с API Replicate для работы с языковыми моделями.
    """
    def __init__(self, api_token: str, model_version: str, prompt: str) -> None:
        """
        Инициализация клиента ReplicateApi.

        Args:
            api_token (str): API токен Replicate.
            model_version (str): Идентификатор версии модели на Replicate.
            prompt (str): Системное сообщение или базовый prompt для модели.
        """
        self.api_token = api_token
        self.model_version = model_version
        self.prompt = prompt
        self.api_url = "https://api.replicate.com/v1/predictions"
        self.headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }

    def query_llm(
        self,
        question: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 500,
        polling_timeout: int = 60
    ) -> str:
        """
        Отправляет запрос к API Replicate и возвращает ответ модели.
        Реализован базовый polling для ожидания завершения предсказания.

        Args:
            question (str): Вопрос или сообщение пользователя.
            temperature (float, optional): Параметр temperature для генерации. По умолчанию 0.2.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.95.
            max_tokens (int, optional): Максимальное число токенов в ответе. По умолчанию 500.
            polling_timeout (int, optional): Таймаут ожидания завершения предсказания в секундах. По умолчанию 60.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        full_prompt = f"{self.prompt}\n\nUser: {question}"
        payload: Dict[str, Any] = {
            "version": self.model_version,
            "input": {
                "prompt": full_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
        }

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            prediction_id = data.get("id")
            if not prediction_id:
                return "Не удалось получить идентификатор предсказания"

            poll_url = f"{self.api_url}/{prediction_id}"
            start_time = time.time()
            while data.get("status") not in ["succeeded", "failed"]:
                if time.time() - start_time > polling_timeout:
                    return "Polling timeout exceeded"
                poll_response = requests.get(poll_url, headers=self.headers, timeout=10)
                poll_response.raise_for_status()
                data = poll_response.json()
                time.sleep(1)

            if data.get("status") == "succeeded":
                output = data.get("output", "")
                if isinstance(output, list):
                    return " ".join(map(str, output))
                return str(output)
            else:
                return f"Prediction failed with status: {data.get('status')}"
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка при выполнении запроса к API Replicate: %s", e)
            return f"Request failed: {e}"
        except (KeyError, IndexError) as e:
            logger.error("Неожиданный формат ответа: %s", e)
            return f"Unexpected response format: {e}"
