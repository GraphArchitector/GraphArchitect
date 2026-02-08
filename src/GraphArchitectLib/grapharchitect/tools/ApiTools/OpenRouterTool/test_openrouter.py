"""
Тесты для OpenRouter интеграции.

Включает:
- Unit тесты (без реальных API вызовов)
- Integration тесты (с реальными API вызовами, если есть ключ)
"""

import pytest
import os
from unittest.mock import Mock, patch

from .openrouter_llm import OpenRouterLLM, OpenRouterTool
from .openrouter_config import OpenRouterConfig, ModelConfig
from .openrouter_basetool import (
    OpenRouterChatTool,
    OpenRouterClassifierTool,
    OpenRouterSummarizerTool,
    OpenRouterAnalyzerTool
)


# ==================== Unit тесты (без API) ====================

class TestOpenRouterConfig:
    """Тесты конфигурации моделей"""
    
    def test_get_model(self):
        """Получение модели по ключу"""
        model = OpenRouterConfig.get_model("gpt-4")
        
        assert model is not None
        assert model.model_id == "openai/gpt-4"
        assert model.provider == "openai"
    
    def test_get_model_id(self):
        """Получение ID модели"""
        model_id = OpenRouterConfig.get_model_id("gpt-3.5-turbo")
        
        assert model_id == "openai/gpt-3.5-turbo"
    
    def test_list_models(self):
        """Список всех моделей"""
        models = OpenRouterConfig.list_models()
        
        assert len(models) > 0
        assert "gpt-4" in models
        assert "claude-3.5-sonnet" in models
    
    def test_list_by_provider(self):
        """Модели конкретного провайдера"""
        openai_models = OpenRouterConfig.list_by_provider("openai")
        
        assert len(openai_models) > 0
        
        for model in openai_models.values():
            assert model.provider == "openai"
    
    def test_get_cheapest_model(self):
        """Самая дешевая модель"""
        cheapest = OpenRouterConfig.get_cheapest_model()
        
        assert cheapest is not None
        assert cheapest.cost_per_1m_tokens == 0.0  # Free tier
    
    def test_get_best_model(self):
        """Лучшая модель"""
        best = OpenRouterConfig.get_best_model()
        
        assert best is not None
        assert "gpt-4" in best.model_id.lower()


class TestOpenRouterLLM:
    """Тесты OpenRouter клиента"""
    
    def test_initialization_with_api_key(self):
        """Инициализация с API ключом"""
        client = OpenRouterLLM(
            api_key="sk-or-test-key",
            model_name="openai/gpt-3.5-turbo"
        )
        
        assert client.api_key == "sk-or-test-key"
        assert client.model_name == "openai/gpt-3.5-turbo"
    
    def test_initialization_without_api_key_fails(self):
        """Инициализация без ключа выбрасывает ошибку"""
        # Временно удаляем env переменную
        old_key = os.environ.pop("OPENROUTER_API_KEY", None)
        
        try:
            with pytest.raises(ValueError, match="API ключ не указан"):
                OpenRouterLLM(api_key=None)
        finally:
            # Восстанавливаем если была
            if old_key:
                os.environ["OPENROUTER_API_KEY"] = old_key
    
    @patch('requests.post')
    def test_query_llm_success(self, mock_post):
        """Успешный запрос к LLM"""
        # Мокаем ответ
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ],
            "usage": {
                "total_tokens": 100
            }
        }
        mock_post.return_value = mock_response
        
        # Запрос
        client = OpenRouterLLM(api_key="test-key")
        result = client.query_llm("Test question")
        
        assert result == "Test response"
        assert mock_post.called
    
    @patch('requests.post')
    def test_query_llm_timeout(self, mock_post):
        """Обработка таймаута"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        client = OpenRouterLLM(api_key="test-key")
        result = client.query_llm("Test question")
        
        assert "timeout" in result.lower()


class TestOpenRouterBaseTool:
    """Тесты BaseTool реализаций"""
    
    def test_chat_tool_creation(self):
        """Создание ChatTool"""
        tool = OpenRouterChatTool(
            api_key="test-key",
            model_key="gpt-3.5-turbo"
        )
        
        assert tool.metadata.tool_name == "OpenRouter-GPT-3.5 Turbo"
        assert tool.input.format == "text|question"
        assert tool.output.format == "text|answer"
    
    def test_classifier_tool_creation(self):
        """Создание ClassifierTool"""
        tool = OpenRouterClassifierTool(
            api_key="test-key",
            categories=["spam", "ham"]
        )
        
        assert tool.input.format == "text|question"
        assert tool.output.format == "text|category"
        assert tool.categories == ["spam", "ham"]
    
    def test_summarizer_tool_creation(self):
        """Создание SummarizerTool"""
        tool = OpenRouterSummarizerTool(
            api_key="test-key",
            max_summary_words=50
        )
        
        assert tool.input.format == "text|document"
        assert tool.output.format == "text|summary"
    
    def test_analyzer_tool_creation(self):
        """Создание AnalyzerTool"""
        tool = OpenRouterAnalyzerTool(
            api_key="test-key",
            analysis_type="sentiment"
        )
        
        assert tool.input.format == "text|data"
        assert tool.output.format == "text|analysis"


# ==================== Integration тесты (требуют API ключ) ====================

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
class TestOpenRouterIntegration:
    """Интеграционные тесты с реальным API"""
    
    def test_real_chat_request(self):
        """Реальный запрос к ChatGPT"""
        tool = OpenRouterChatTool(model_key="gpt-3.5-turbo")
        
        result = tool.execute("What is 2+2? Answer with just the number.")
        
        assert "4" in result
    
    def test_real_classification(self):
        """Реальная классификация"""
        tool = OpenRouterClassifierTool(
            model_key="gpt-3.5-turbo",
            categories=["positive", "negative"]
        )
        
        result = tool.execute("This is absolutely amazing!")
        
        assert result.lower() in ["positive", "negative"]
    
    def test_different_models(self):
        """Тест разных моделей"""
        models = ["gpt-3.5-turbo", "llama-3-8b"]
        question = "What is AI?"
        
        for model_key in models:
            try:
                tool = OpenRouterChatTool(model_key=model_key)
                result = tool.execute(question)
                
                assert len(result) > 0
                print(f"✓ {model_key}: {len(result)} chars")
            
            except Exception as e:
                print(f"⚠️ {model_key} failed: {e}")


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])
