import requests
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChatGPTVisualApi:
    """
    Клиент для взаимодействия с ChatGPT API (например, GPT-4o) для задач описания изображений и VisualQA.
    """
    def __init__(self, api_key: str, model_name: str, system_prompt: str) -> None:
        """
        Инициализация клиента ChatGPTVisualApi.

        Args:
            api_key (str): API ключ OpenAI.
            model_name (str): Имя модели (например, 'gpt-4o' или 'gpt-4-vision').
            system_prompt (str): Системное сообщение (начальный контекст) для модели.
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
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 500,
        timeout: int = 10
    ) -> str:
        """
        Отправляет запрос к ChatGPT API и возвращает ответ модели.

        Args:
            messages (List[Dict[str, str]]): Список сообщений в формате [{"role": "system/user", "content": "..."}, ...].
            temperature (float, optional): Параметр temperature для сэмплирования. По умолчанию 0.2.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.95.
            max_tokens (int, optional): Максимальное число токенов в ответе. По умолчанию 500.
            timeout (int, optional): Таймаут запроса в секундах. По умолчанию 10.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
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

    def describe_image(
        self,
        image_url: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 500,
        timeout: int = 10
    ) -> str:
        """
        Получает описание изображения, используя модель с поддержкой визуального ввода.

        Args:
            image_url (str): URL изображения для описания.
            temperature (float, optional): Параметр temperature для генерации. По умолчанию 0.2.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.95.
            max_tokens (int, optional): Максимальное число токенов в ответе. По умолчанию 500.
            timeout (int, optional): Таймаут запроса в секундах. По умолчанию 10.

        Returns:
            str: Описание изображения или сообщение об ошибке.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            # Передаём URL изображения вместе с инструкцией описать его.
            {"role": "user", "content": f"Image: {image_url}\nОпиши данное изображение."}
        ]
        return self.query_llm(messages, temperature, top_p, max_tokens, timeout)

    def answer_visual_question(
        self,
        image_url: str,
        question: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 500,
        timeout: int = 10
    ) -> str:
        """
        Отвечает на вопрос, связанный с изображением, используя модель с поддержкой визуального ввода.

        Args:
            image_url (str): URL изображения для анализа.
            question (str): Вопрос по изображению.
            temperature (float, optional): Параметр temperature для генерации. По умолчанию 0.2.
            top_p (float, optional): Параметр nucleus sampling. По умолчанию 0.95.
            max_tokens (int, optional): Максимальное число токенов в ответе. По умолчанию 500.
            timeout (int, optional): Таймаут запроса в секундах. По умолчанию 10.

        Returns:
            str: Ответ модели на заданный вопрос или сообщение об ошибке.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            # Передаём URL изображения и вопрос для анализа.
            {"role": "user", "content": f"Image: {image_url}\nQuestion: {question}"}
        ]
        return self.query_llm(messages, temperature, top_p, max_tokens, timeout)
