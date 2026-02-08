"""
Клиент для работы с Fractal Agents RAG API.
Предоставляет методы для управления индексами и документами.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class APIResponse:
    """Базовый класс для ответов API."""
    success: bool
    status_code: int
    raw_response: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class IndexCreateResponse(APIResponse):
    """Ответ на создание индекса."""
    index_name: Optional[str] = None


@dataclass
class DocumentUploadResponse(APIResponse):
    """Ответ на загрузку документа."""
    document_hash: Optional[str] = None


@dataclass
class DocumentRegion:
    """Регион документа с координатами."""
    document_name: str
    page_number: int
    bottom_left_x: float
    bottom_left_y: float
    top_right_x: float
    top_right_y: float


@dataclass
class RelevantDocument:
    """Релевантный документ из результатов поиска."""
    document_name: str
    document_hash: str
    download_url: str
    regions: List[DocumentRegion]


@dataclass
class Snippet:
    """Фрагмент текста из документа."""
    document_hash: str
    snippet_text: str
    lemmatize_text: str
    processed_text: str
    block_size: int
    metadata: Dict[str, Any]


@dataclass
class AnswerResponse(APIResponse):
    """Ответ на запрос к документам."""
    index_name: Optional[str] = None
    answer: Optional[str] = None
    relevant_documents: List[RelevantDocument] = None
    snippets: List[Snippet] = None
    hallucination_probability: Optional[float] = None

    def __post_init__(self):
        """Инициализация списков по умолчанию."""
        if self.relevant_documents is None:
            self.relevant_documents = []
        if self.snippets is None:
            self.snippets = []


@dataclass
class DeleteResponse(APIResponse):
    """Ответ на удаление индекса или документа."""
    pass


@dataclass
class DocumentsCountResponse(APIResponse):
    """Ответ на запрос количества документов."""
    count: Optional[int] = None


class FractalAgentsClient:
    """
    Клиент для работы с Fractal Agents RAG API.
    
    Attributes:
        api_key: API ключ для аутентификации
        base_url: Базовый URL API
    """
    
    BASE_URL = "https://api.fractalagents.ai/rag/api"
    
    def __init__(self, api_key: str):
        """
        Инициализация клиента.
        
        Args:
            api_key: API ключ для аутентификации
        """
        self.api_key = api_key
        self.base_url = self.BASE_URL
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Получить заголовки для запросов.
        
        Returns:
            Словарь с заголовками
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _handle_response(
        self,
        response: requests.Response,
        response_class: type
    ) -> APIResponse:
        """
        Обработка ответа от API.
        
        Args:
            response: Ответ от requests
            response_class: Класс для создания ответа
            
        Returns:
            Экземпляр соответствующего класса ответа
        """
        try:
            json_data = response.json()
        except ValueError:
            json_data = {"raw_text": response.text}
        
        success = json_data.get("success", False)
        error_message = json_data.get("error_message")
        
        if response_class == IndexCreateResponse:
            return IndexCreateResponse(
                success=success,
                status_code=response.status_code,
                raw_response=json_data,
                error_message=error_message,
                index_name=json_data.get("index_name")
            )
        
        elif response_class == DocumentUploadResponse:
            return DocumentUploadResponse(
                success=success,
                status_code=response.status_code,
                raw_response=json_data,
                error_message=error_message,
                document_hash=json_data.get("document_hash")
            )
        
        elif response_class == AnswerResponse:
            # Парсинг релевантных документов
            relevant_docs = []
            for doc in json_data.get("relevant_documents", []):
                regions = [
                    DocumentRegion(
                        document_name=reg.get("documentName"),
                        page_number=reg.get("pageNumber"),
                        bottom_left_x=reg.get("bottomLeftX"),
                        bottom_left_y=reg.get("bottomLeftY"),
                        top_right_x=reg.get("topRightX"),
                        top_right_y=reg.get("topRightY")
                    )
                    for reg in doc.get("regions", [])
                ]
                
                relevant_docs.append(RelevantDocument(
                    document_name=doc.get("document_name"),
                    document_hash=doc.get("document_hash"),
                    download_url=doc.get("download_url"),
                    regions=regions
                ))
            
            # Парсинг фрагментов
            snippets = [
                Snippet(
                    document_hash=snip.get("documentHash"),
                    snippet_text=snip.get("snippetText"),
                    lemmatize_text=snip.get("lemmatizeText"),
                    processed_text=snip.get("processedText"),
                    block_size=snip.get("blockSize"),
                    metadata=snip.get("metadata", {})
                )
                for snip in json_data.get("snippets", [])
            ]
            
            return AnswerResponse(
                success=success,
                status_code=response.status_code,
                raw_response=json_data,
                error_message=error_message,
                index_name=json_data.get("index_name"),
                answer=json_data.get("answer"),
                relevant_documents=relevant_docs,
                snippets=snippets,
                hallucination_probability=json_data.get("hallucination_probability")
            )
        
        elif response_class == DeleteResponse:
            return DeleteResponse(
                success=success,
                status_code=response.status_code,
                raw_response=json_data,
                error_message=error_message
            )
        
        elif response_class == DocumentsCountResponse:
            return DocumentsCountResponse(
                success=success,
                status_code=response.status_code,
                raw_response=json_data,
                error_message=error_message,
                count=json_data.get("count")
            )
        
        else:
            return APIResponse(
                success=success,
                status_code=response.status_code,
                raw_response=json_data,
                error_message=error_message
            )
    
    def create_index(self, index_name: str) -> IndexCreateResponse:
        """
        Создать новый индекс.
        
        Args:
            index_name: Название индекса
            
        Returns:
            IndexCreateResponse с результатом операции
        """
        url = f"{self.base_url}/create-index"
        payload = {"index_name": index_name}
        
        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        
        return self._handle_response(response, IndexCreateResponse)
    
    def upload_document(
        self,
        index_name: str,
        file_path: str,
        document_name: Optional[str] = None
    ) -> DocumentUploadResponse:
        """
        Загрузить документ в индекс.
        
        Args:
            index_name: Название индекса
            file_path: Путь к файлу
            document_name: Имя документа (если не указано, используется имя файла)
            
        Returns:
            DocumentUploadResponse с результатом операции
        """
        url = f"{self.base_url}/upload-file"
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if document_name is None:
            document_name = path.name
        
        with open(file_path, 'rb') as f:
            file_content = list(f.read())
        
        payload = {
            'index_name': index_name,
            'document_name': document_name,
            'data': file_content
        }
        
        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        
        return self._handle_response(response, DocumentUploadResponse)
    
    def get_answer(
        self,
        index_name: str,
        query: str,
        think: bool = False,
        deepthink: bool = False
    ) -> AnswerResponse:
        """
        Получить ответ на запрос по документам в индексе.
        
        Args:
            index_name: Название индекса
            query: Запрос
            think: Использовать режим think
            deepthink: Использовать режим deepthink
            
        Returns:
            AnswerResponse с ответом и релевантными документами
        """
        url = f"{self.base_url}/get-answer"
        
        payload = {
            "query": query,
            "index_name": index_name,
            "think": think,
            "deepthink": deepthink
        }
        
        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload
        )
        
        return self._handle_response(response, AnswerResponse)
    
    def delete_index(self, index_name: str) -> DeleteResponse:
        """
        Удалить индекс.
        
        Args:
            index_name: Название индекса
            
        Returns:
            DeleteResponse с результатом операции
        """
        url = f"{self.base_url}/delete-index"
        payload = {"index_name": index_name}
        
        response = requests.delete(
            url,
            headers=self._get_headers(),
            json=payload
        )
        
        return self._handle_response(response, DeleteResponse)
    
    def delete_document(
        self,
        index_name: str,
        document_hash: str
    ) -> DeleteResponse:
        """
        Удалить документ из индекса.
        
        Args:
            index_name: Название индекса
            document_hash: Хэш документа
            
        Returns:
            DeleteResponse с результатом операции
        """
        url = f"{self.base_url}/delete-document"
        
        payload = {
            "index_name": index_name,
            "document_hash": document_hash
        }
        
        response = requests.delete(
            url,
            headers=self._get_headers(),
            json=payload
        )
        
        return self._handle_response(response, DeleteResponse)
    
    def get_documents_count(self, index_name: str) -> DocumentsCountResponse:
        """
        Получить количество документов в индексе.
        
        Args:
            index_name: Название индекса
            
        Returns:
            DocumentsCountResponse с количеством документов
        """
        url = f"{self.base_url}/get-documents-count"
        
        params = {"index_name": index_name}
        
        response = requests.get(
            url,
            headers=self._get_headers(),
            params=params
        )
        
        return self._handle_response(response, DocumentsCountResponse)