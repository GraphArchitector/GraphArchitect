import requests
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AudioReplicateApi:
    """
    Клиент для взаимодействия с API Replicate для работы со звуком.
    """
    def __init__(self, api_token: str, model_version: str, base_prompt: str = "") -> None:
        """
        Инициализация клиента для звуковых задач.

        Args:
            api_token (str): API токен Replicate.
            model_version (str): Идентификатор версии модели на Replicate.
            base_prompt (str, optional): Базовый prompt для модели, задающий общий контекст.
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
        Вспомогательный метод для опроса статуса предсказания.

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

    def _send_request(self, payload: Dict[str, Any], polling_timeout: int) -> str:
        """
        Вспомогательный метод для отправки запроса и получения результата.

        Args:
            payload (Dict[str, Any]): Полезная нагрузка для запроса.
            polling_timeout (int): Таймаут для опроса статуса.

        Returns:
            str: Ответ модели или сообщение об ошибке.
        """
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            prediction_id = data.get("id")
            if not prediction_id:
                return "Не удалось получить идентификатор предсказания"
            final_data = self._poll_prediction(prediction_id, polling_timeout)
            if final_data.get("status") == "succeeded":
                output = final_data.get("output", "")
                if isinstance(output, list):
                    return " ".join(map(str, output))
                return str(output)
            else:
                return f"Prediction failed with status: {final_data.get('status')}"
        except requests.exceptions.RequestException as e:
            logger.error("Ошибка запроса к API Replicate: %s", e)
            return f"Request failed: {e}"
        except (KeyError, IndexError) as e:
            logger.error("Неожиданный формат ответа: %s", e)
            return f"Unexpected response format: {e}"
        except TimeoutError as te:
            logger.error(te)
            return str(te)

    def transcribe_audio(self, audio_url: str, language: str = "ru", polling_timeout: int = 60) -> str:
        """
        Транскрибирует аудио по заданному URL.

        Args:
            audio_url (str): URL аудиофайла для транскрипции.
            language (str, optional): Язык аудио. По умолчанию "ru".
            polling_timeout (int, optional): Таймаут ожидания в секундах.

        Returns:
            str: Транскрипция аудио или сообщение об ошибке.
        """
        prompt = (
            f"{self.base_prompt}\n\n"
            f"Audio: {audio_url}\n"
            f"Язык: {language}\n"
            f"Транскрибируй аудио."
        )
        payload: Dict[str, Any] = {
            "version": self.model_version,
            "input": {
                "prompt": prompt
            }
        }
        return self._send_request(payload, polling_timeout)

    def answer_audio_question(self, audio_url: str, question: str, polling_timeout: int = 60) -> str:
        """
        Отвечает на вопрос, касающийся содержания аудио.

        Args:
            audio_url (str): URL аудиофайла для анализа.
            question (str): Вопрос по содержимому аудио.
            polling_timeout (int, optional): Таймаут ожидания в секундах.

        Returns:
            str: Ответ модели на заданный вопрос или сообщение об ошибке.
        """
        prompt = (
            f"{self.base_prompt}\n\n"
            f"Audio: {audio_url}\n"
            f"Question: {question}"
        )
        payload: Dict[str, Any] = {
            "version": self.model_version,
            "input": {
                "prompt": prompt
            }
        }
        return self._send_request(payload, polling_timeout)
