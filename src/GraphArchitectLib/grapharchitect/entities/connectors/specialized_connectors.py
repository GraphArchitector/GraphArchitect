"""
Специализированные коннекторы из отчета.

Реализует 10+ типов коннекторов для специфических задач:
- Разбор песен
- Графики физических процессов
- Медицинские изображения (МРТ)
- Картографические данные
И другие
"""

from typing import Optional, Dict, Any
from .connector import Connector


class SpecializedConnectorFactory:
    """Фабрика для создания специализированных коннекторов."""
    
    @staticmethod
    def create_song_analysis_connector() -> Connector:
        """
        Разбор песни.
        
        Выходной коннектор для инструментов создающих отчеты 
        с подробным разбором текстов песен.
        
        Формат: Структурированный текст отчета на русском языке (~5000 символов)
        """
        connector = Connector("text", "report")
        connector.properties = {
            "language": "russian",
            "encoding": ["UTF-8", "Windows-1251", "KOI8-R"],
            "length": 5000,
            "style": "simplified",
            "knowledge_domain": "general",
            "template": "structured_text",
            "content_type": "song_analysis"
        }
        return connector
    
    @staticmethod
    def create_physics_graph_connector() -> Connector:
        """
        Разбор графика физического процесса.
        
        Выходной коннектор для инструментов создающих отчеты
        с разбором графиков физических процессов.
        
        Формат: Технический отчет (100-10000 символов)
        """
        connector = Connector("text", "report")
        connector.properties = {
            "language": "russian",
            "encoding": ["UTF-8", "Windows-1251", "KOI8-R"],
            "length_range": [100, 10000],
            "style": "technical",
            "knowledge_domain": "physics",
            "template": "graph_analysis",
            "content_type": "physics_process"
        }
        return connector
    
    @staticmethod
    def create_biology_reasoning_connector() -> Connector:
        """
        Размышление на биологическую тему.
        
        Выходной коннектор для инструментов, предоставляющих 
        ответы на вопросы по биологии.
        
        Формат: Научный диалог (100-10000 символов)
        """
        connector = Connector("text", "answer")
        connector.properties = {
            "language": "russian",
            "encoding": ["UTF-8", "Windows-1251", "KOI8-R"],
            "length_range": [100, 10000],
            "style": "scientific",
            "format": "dialog",
            "knowledge_domain": "biology",
            "content_type": "reasoning"
        }
        return connector
    
    @staticmethod
    def create_math_reasoning_connector() -> Connector:
        """
        Размышление на тему математики.
        
        Выходной коннектор для ответов на математические вопросы.
        
        Формат: Проверка гипотез (1000-3000 символов)
        """
        connector = Connector("text", "reasoning")
        connector.properties = {
            "language": "russian",
            "encoding": ["UTF-8", "Windows-1251", "KOI8-R"],
            "length_range": [1000, 3000],
            "reasoning_type": "hypothesis_verification",
            "knowledge_domain": "math",
            "content_type": "mathematical_reasoning"
        }
        return connector
    
    @staticmethod
    def create_math_audio_description_connector() -> Connector:
        """
        Описание математической темы (аудио).
        
        Выходной коннектор для голосового объяснения математических тем.
        
        Формат: Аудио речь на русском (1000-3000 символов текста)
        """
        connector = Connector("audio", "speech")
        connector.properties = {
            "language": "russian",
            "text_length_range": [1000, 3000],
            "tempo": "medium",
            "knowledge_domain": "math",
            "content_type": "audio_explanation"
        }
        return connector
    
    @staticmethod
    def create_song_narration_connector() -> Connector:
        """
        Повествование (песня, аудио).
        
        Выходной коннектор для голосовой реконструкции музыкального текста.
        
        Формат: Аудио (5 сек, 2 канала, 16 бит, 44100 Гц)
        """
        connector = Connector("audio", "speech")
        connector.properties = {
            "duration_seconds": 5,
            "channels": 2,
            "bit_depth": 16,
            "sample_rate": 44100,
            "style": "narration",
            "knowledge_domain": "general",
            "template": "structured_text",
            "content_type": "song_narration"
        }
        return connector
    
    @staticmethod
    def create_materials_definition_audio_connector() -> Connector:
        """
        Определение в звуковом формате.
        
        Выходной коннектор для определений, связанных с наукой о материалах.
        
        Формат: Аудио (15 сек, моно, 24 бит, 48000 Гц)
        """
        connector = Connector("audio", "speech")
        connector.properties = {
            "duration_seconds": 15,
            "channels": 1,
            "bit_depth": 24,
            "sample_rate": 48000,
            "style": "simplified",
            "template": "text",
            "knowledge_domain": "general",
            "content_type": "materials_definition"
        }
        return connector
    
    @staticmethod
    def create_general_speech_english_connector() -> Connector:
        """
        Общая речь (английский).
        
        Выходной коннектор для обработки общей речи на английском.
        
        Формат: Аудио (10 сек, 6 каналов, 32 бит, 96000 Гц)
        """
        connector = Connector("audio", "speech")
        connector.properties = {
            "duration_seconds": 10,
            "channels": 6,
            "bit_depth": 32,
            "sample_rate": 96000,
            "language": "english",
            "tempo": "fast",
            "knowledge_domain": "general",
            "content_type": "general_speech"
        }
        return connector
    
    @staticmethod
    def create_data_table_image_connector() -> Connector:
        """
        Данные таблицы (изображение).
        
        Выходной коннектор для научных графиков.
        
        Формат: Изображение (100-1000 × 100-300)
        """
        connector = Connector("image", "report")
        connector.properties = {
            "resolution_range": [[100, 1000], [100, 300]],
            "content": "table",
            "knowledge_domain": "biology",
            "content_type": "scientific_graph"
        }
        return connector
    
    @staticmethod
    def create_painting_connector() -> Connector:
        """
        Картина.
        
        Выходной коннектор для сгенерированных изображений.
        
        Формат: RGB/grayscale, разные разрешения
        """
        connector = Connector("image", "raw")
        connector.properties = {
            "color": ["RGB", "grayscale"],
            "pixel_type": ["uint8", "uint16", "float32"],
            "resolutions": ["640x480", "1280x720", "1920x1080"],
            "content": "drawing",
            "knowledge_domain": "general",
            "content_type": "painting"
        }
        return connector
    
    @staticmethod
    def create_mri_connector() -> Connector:
        """
        МРТ изображение.
        
        Выходной коннектор для анализа медицинских МРТ изображений.
        
        Формат: RGB изображение с анализом
        """
        connector = Connector("image", "reasoning")
        connector.properties = {
            "color": "RGB",
            "pixel_type": ["uint8", "uint16", "float32"],
            "resolutions": ["640x480", "1280x720", "1920x1080"],
            "reasoning_type": "hypothesis_verification",
            "content": "MRI",
            "knowledge_domain": "medicine",
            "content_type": "medical_imaging"
        }
        return connector
    
    @staticmethod
    def create_map_image_connector() -> Connector:
        """
        Изображение карты.
        
        Выходной коннектор для интерпретации географических изображений.
        
        Формат: RGB изображение карты
        """
        connector = Connector("image", "raw")
        connector.properties = {
            "color": "RGB",
            "pixel_type": ["uint8", "uint16", "float32"],
            "resolutions": ["640x480", "1280x720", "1920x1080"],
            "content": "location",
            "knowledge_domain": "general",
            "content_type": "map"
        }
        return connector
    
    @staticmethod
    def get_all_specialized_connectors() -> Dict[str, Connector]:
        """
        Получить все специализированные коннекторы.
        
        Returns:
            Словарь {название: коннектор}
        """
        factory = SpecializedConnectorFactory
        
        return {
            "song_analysis": factory.create_song_analysis_connector(),
            "physics_graph": factory.create_physics_graph_connector(),
            "biology_reasoning": factory.create_biology_reasoning_connector(),
            "math_reasoning": factory.create_math_reasoning_connector(),
            "math_audio": factory.create_math_audio_description_connector(),
            "song_narration": factory.create_song_narration_connector(),
            "materials_audio": factory.create_materials_definition_audio_connector(),
            "speech_english": factory.create_general_speech_english_connector(),
            "data_table": factory.create_data_table_image_connector(),
            "painting": factory.create_painting_connector(),
            "mri": factory.create_mri_connector(),
            "map": factory.create_map_image_connector()
        }
