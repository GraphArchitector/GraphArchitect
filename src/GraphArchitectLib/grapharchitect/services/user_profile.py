"""
Профиль пользователя для адаптации системы.

Позволяет GraphArchitect настраиваться под конкретного пользователя:
- Приоритеты (скорость vs качество vs стоимость)
- История взаимодействий
- Предпочтительные инструменты
- Области знаний
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class UserPriority(Enum):
    """Приоритет пользователя."""
    SPEED = "speed"          # Максимальная скорость
    QUALITY = "quality"      # Максимальное качество
    COST = "cost"            # Минимальная стоимость
    BALANCED = "balanced"    # Баланс всех параметров


@dataclass
class UserProfile:
    """
    Профиль пользователя.
    
    Содержит предпочтения, которые влияют на:
    - Веса функций качества (w_q, w_c, w_t)
    - Выбор алгоритма планирования
    - Температуру (exploration vs exploitation)
    - Предпочтительные инструменты
    """
    
    user_id: str = "default"
    
    # Основной приоритет
    priority: UserPriority = UserPriority.BALANCED
    
    # Веса для формулы R(x) (настраиваются автоматически)
    weight_quality: float = 1.0
    weight_cost: float = 0.3
    weight_time: float = 0.2
    
    # Температура (exploration vs exploitation)
    temperature_multiplier: float = 1.0  # <1 = exploitation, >1 = exploration
    
    # Максимальные ограничения
    max_cost_per_task: Optional[float] = None    # Максимальная стоимость задачи
    max_time_per_task: Optional[float] = None    # Максимальное время (секунды)
    max_steps: Optional[int] = None              # Максимальное количество шагов
    
    # Предпочтительный алгоритм
    preferred_algorithm: str = "yen_5"
    
    # Области знаний (для более точного NLI)
    knowledge_domains: List[str] = field(default_factory=lambda: ["general"])
    
    # Язык
    language: str = "russian"
    
    # Статистика пользователя
    total_tasks: int = 0
    avg_satisfaction: float = 0.5
    
    # История предпочтений (какие инструменты нравились)
    preferred_tools: Dict[str, float] = field(default_factory=dict)
    
    def apply_priority(self):
        """Применить приоритет к весам."""
        if self.priority == UserPriority.SPEED:
            self.weight_quality = 0.5
            self.weight_cost = 0.2
            self.weight_time = 2.0  # Максимизировать скорость
            self.temperature_multiplier = 0.5  # Быстрый выбор
        
        elif self.priority == UserPriority.QUALITY:
            self.weight_quality = 2.0  # Максимизировать качество
            self.weight_cost = 0.1
            self.weight_time = 0.1
            self.temperature_multiplier = 0.3  # Выбирать лучших
        
        elif self.priority == UserPriority.COST:
            self.weight_quality = 0.5
            self.weight_cost = 2.0  # Минимизировать стоимость
            self.weight_time = 0.3
            self.temperature_multiplier = 0.7
        
        elif self.priority == UserPriority.BALANCED:
            self.weight_quality = 1.0
            self.weight_cost = 0.5
            self.weight_time = 0.5
            self.temperature_multiplier = 1.0
    
    def update_from_feedback(self, task_type: str, tool_name: str, score: float):
        """
        Обновить профиль на основе обратной связи.
        
        Args:
            task_type: Тип задачи
            tool_name: Использованный инструмент
            score: Оценка удовлетворенности (0-1)
        """
        self.total_tasks += 1
        
        # Обновляем среднюю удовлетворенность
        alpha = 0.9
        self.avg_satisfaction = alpha * self.avg_satisfaction + (1 - alpha) * score
        
        # Обновляем предпочтения по инструментам
        old_pref = self.preferred_tools.get(tool_name, 0.5)
        self.preferred_tools[tool_name] = alpha * old_pref + (1 - alpha) * score
    
    def get_tool_bonus(self, tool_name: str) -> float:
        """
        Получить бонус к логиту для предпочитаемого инструмента.
        
        Args:
            tool_name: Название инструмента
            
        Returns:
            Бонус (может быть положительным или отрицательным)
        """
        pref = self.preferred_tools.get(tool_name, 0.5)
        # Бонус: от -0.5 до +0.5
        return (pref - 0.5) * 1.0


class UserProfileManager:
    """
    Менеджер профилей пользователей.
    
    Хранит и управляет профилями.
    """
    
    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
        self._default_profile = UserProfile()
    
    def get_profile(self, user_id: str) -> UserProfile:
        """Получить профиль пользователя (или создать дефолтный)."""
        if user_id not in self._profiles:
            profile = UserProfile(user_id=user_id)
            self._profiles[user_id] = profile
            logger.info(f"Created new profile for user: {user_id}")
        
        return self._profiles[user_id]
    
    def set_priority(self, user_id: str, priority: UserPriority):
        """Установить приоритет пользователя."""
        profile = self.get_profile(user_id)
        profile.priority = priority
        profile.apply_priority()
        
        logger.info(f"User {user_id} priority set to: {priority.value}")
    
    def get_selector_params(self, user_id: str) -> Dict:
        """
        Получить параметры для InstrumentSelector на основе профиля.
        
        Returns:
            Словарь с параметрами
        """
        profile = self.get_profile(user_id)
        
        return {
            "weight_quality": profile.weight_quality,
            "weight_cost": profile.weight_cost,
            "weight_time": profile.weight_time,
            "temperature_multiplier": profile.temperature_multiplier,
            "max_cost": profile.max_cost_per_task,
            "max_time": profile.max_time_per_task,
            "max_steps": profile.max_steps,
            "preferred_algorithm": profile.preferred_algorithm
        }
