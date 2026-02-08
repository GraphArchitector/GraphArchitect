import requests
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FunctionCallingReplicateApi:
    """
    Клиент для взаимодействия с API Replicate с поддержкой механизма function calling.
    Предполагается, что модель поддерживает передачу описаний функций в запросе.
    """

    def __init__(self, api_token: str, model_version: str, base_prompt: str = "") -> None:
        """
        Инициализация клиента для работы с функциями.

        Args:
            api_token (str): API токен Replicate.
            model_version (str): Идентификатор версии модели.
            base_prompt (str, optional): Базовый контекст для модели.
        """
        self.api_token = api_token
        self.model_version = model_version
        self.base_prompt = base_prompt
        self.api_url = "https://api.replicate.com/v1/predictions"
        self.headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json"
        }

    def _poll_prediction(self, prediction_id: str, polling_timeout: int = 60) -> Dict[str, Any]:
        """
        Вспомогательный метод для опроса статуса предсказания до завершения.

        Args:
            prediction_id (str): Идентификатор предсказания.
            polling_timeout (int, optional): Таймаут ожидания в секундах.

        Returns:
            Dict[str, Any]: Финальный ответ от API.
        """
        poll_url = f"{self.api_url}/{prediction_id}"
        start_time = time.time()
        data = {}
        while True:
            if time.time() - start_time > polling_timeout:
                raise TimeoutError("Polling timeout exceeded")
            poll_response = requests.get(poll_url, headers=self.headers, timeout=10)
            poll_response.raise_for_status()
            data = poll_response.json()
            if data.get("status") in ["succeeded", "failed"]:
                break
            time.sleep(1)
        return data

    def _send_request(self, payload: Dict[str, Any], polling_timeout: int) -> Dict[str, Any]:
        """
        Отправляет запрос к API Replicate и возвращает финальный ответ.

        Args:
            payload (Dict[str, Any]): Полезная нагрузка для запроса.
            polling_timeout (int): Таймаут для опроса.

        Returns:
            Dict[str, Any]: Ответ модели.
        """
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            prediction_id = data.get("id")
            if not prediction_id:
                raise ValueError("Не удалось получить идентификатор предсказания")
            final_data = self._poll_prediction(prediction_id, polling_timeout)
            return final_data
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка запроса к API Replicate: %s", e)
            raise e
        except (KeyError, IndexError) as e:
            logger.error("Неожиданный формат ответа: %s", e)
            raise e

    def query_llm(
            self,
            messages: List[Dict[str, str]],
            functions: List[Dict[str, Any]] = None,
            temperature: float = 0.2,
            top_p: float = 0.95,
            max_tokens: int = 500,
            polling_timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Отправляет запрос с сообщениями и описанием функций.

        Args:
            messages (List[Dict[str, str]]): Список сообщений в формате
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}].
            functions (List[Dict[str, Any]], optional): Описание функций для вызова.
            temperature (float, optional): Параметр temperature.
            top_p (float, optional): Параметр nucleus sampling.
            max_tokens (int, optional): Максимальное число токенов.
            polling_timeout (int, optional): Таймаут опроса.

        Returns:
            Dict[str, Any]: Ответ модели, включая данные о вызове функции, если таковой произошёл.
        """
        payload: Dict[str, Any] = {
            "version": self.model_version,
            "input": {
                "prompt": self.base_prompt,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        }
        # Если описания функций заданы, добавляем их в запрос
        if functions:
            payload["input"]["functions"] = functions

        final_data = self._send_request(payload, polling_timeout)
        return final_data

    def process_function_call(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные вызова функции из ответа модели.

        Args:
            response_data (Dict[str, Any]): Ответ модели.

        Returns:
            Dict[str, Any]: Словарь с информацией о вызове функции или пустой словарь.
        """
        # Предполагается, что информация о вызове функции находится в поле "function_call"
        message = response_data.get("output", {})
        if isinstance(message, dict) and "function_call" in message:
            return message["function_call"]
        return {}

    def transcribe_audio(self, audio_url: str, language: str = "ru", polling_timeout: int = 60) -> Dict[str, Any]:
        """
        Пример вызова функции для транскрипции аудио.

        Args:
            audio_url (str): URL аудиофайла.
            language (str, optional): Язык аудио. По умолчанию "ru".
            polling_timeout (int, optional): Таймаут опроса.

        Returns:
            Dict[str, Any]: Ответ модели, включая вызов функции или текст транскрипции.
        """
        # Определяем описание функции для транскрипции
        function_definitions = [
            {
                "name": "transcribe_audio",
                "description": "Транскрибирует аудио и возвращает текст.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_url": {
                            "type": "string",
                            "description": "URL аудиофайла."
                        },
                        "language": {
                            "type": "string",
                            "description": "Язык аудио."
                        }
                    },
                    "required": ["audio_url", "language"]
                }
            }
        ]

        messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user",
             "content": f"Audio: {audio_url}\nЯзык: {language}\nПожалуйста, транскрибируй аудио с помощью функции."}
        ]

        response_data = self.query_llm(messages, functions=function_definitions, polling_timeout=polling_timeout)
        return response_data

    def answer_audio_question(self, audio_url: str, question: str, polling_timeout: int = 60) -> Dict[str, Any]:
        """
        Пример вызова функции для ответа на вопрос по содержимому аудио.

        Args:
            audio_url (str): URL аудиофайла.
            question (str): Вопрос по содержимому аудио.
            polling_timeout (int, optional): Таймаут опроса.

        Returns:
            Dict[str, Any]: Ответ модели, включая данные о вызове функции или текст ответа.
        """
        function_definitions = [
            {
                "name": "answer_audio_question",
                "description": "Отвечает на вопрос по содержимому аудио.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_url": {
                            "type": "string",
                            "description": "URL аудиофайла."
                        },
                        "question": {
                            "type": "string",
                            "description": "Вопрос по содержимому аудио."
                        }
                    },
                    "required": ["audio_url", "question"]
                }
            }
        ]

        messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user",
             "content": f"Audio: {audio_url}\nQuestion: {question}\nПожалуйста, ответь, используя функцию."}
        ]

        response_data = self.query_llm(messages, functions=function_definitions, polling_timeout=polling_timeout)
        return response_data
