"""
ReWOO Planner - Reasoning Without Observation.

Подход ReWOO:
1. Получаем цепочки от графового алгоритма
2. Отправляем в LLM для создания детального плана
3. Выполняем по плану (без промежуточных наблюдений)

Используется Gemini 3 Flash для быстрого планирования.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ReWOOStep:
    """Шаг в ReWOO плане."""
    
    step_id: str
    tool_name: str
    description: str
    depends_on: List[str]  # ID шагов, от которых зависит
    expected_output: str


@dataclass
class ReWOOPlan:
    """Полный план ReWOO."""
    
    steps: List[ReWOOStep]
    reasoning: str  # Обоснование плана от LLM
    estimated_time: float
    estimated_cost: float


class ReWOOPlanner:
    """
    ReWOO планировщик.
    
    Использует LLM для создания детального плана выполнения
    на основе цепочек от графового алгоритма.
    """
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        fallback_to_openrouter: bool = True
    ):
        """
        Инициализация планировщика.
        
        Args:
            gemini_api_key: API ключ Gemini (или GEMINI_API_KEY из env)
            model: Модель (gemini-1.5-flash рекомендуется)
            fallback_to_openrouter: Использовать OpenRouter если Gemini недоступен
        """
        self._gemini_key = gemini_api_key
        self._model = model
        self._fallback_to_openrouter = fallback_to_openrouter
        self._llm = None
        
        # Инициализация LLM
        self._init_llm()
        
        logger.info(f"ReWOO Planner initialized with {model}")
    
    def _init_llm(self):
        """Инициализация LLM для планирования."""
        
        import os
        
        # Claude Sonnet 4.5 для ReWOO планирования (лучшее качество)
        try:
            from ...tools.ApiTools.OpenRouterTool.openrouter_llm import OpenRouterLLM
            
            self._llm = OpenRouterLLM(
                model_name="anthropic/claude-sonnet-4.5",
                system_prompt=(
                    "Ты эксперт-планировщик задач. "
                    "Разбиваешь сложные задачи на подзадачи. "
                    "Всегда отвечай ТОЛЬКО валидным JSON без пояснений."
                )
            )
            
            logger.info("ReWOO using Claude Sonnet 4.5 via OpenRouter")
        
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")
            
            if self._fallback_to_openrouter:
                # Fallback на Gemini Flash
                try:
                    from ...tools.ApiTools.OpenRouterTool.openrouter_llm import OpenRouterLLM
                    
                    self._llm = OpenRouterLLM(
                        model_name="google/gemini-2.5-flash",
                        system_prompt=(
                            "Ты эксперт-планировщик. "
                            "Всегда отвечай ТОЛЬКО JSON без пояснений."
                        )
                    )
                    
                    logger.warning("Fallback to Gemini 2.5 Flash for ReWOO planning")
                except:
                    pass
    
    def create_plan(
        self,
        task_description: str,
        strategies: List[List],  # Цепочки от графового алгоритма
        algorithm_used: str,
        top_tools: Optional[List] = None  # Топ-10 лучших инструментов
    ) -> Optional[ReWOOPlan]:
        """
        Создать детальный план выполнения.
        
        Args:
            task_description: Описание задачи
            strategies: Цепочки инструментов от графового алгоритма
            algorithm_used: Использованный алгоритм (yen_5, dijkstra, etc.)
            
        Returns:
            ReWOOPlan или None
        """
        if not strategies:
            logger.error("No strategies provided")
            return None
        
        if not self._llm:
            logger.warning("LLM not available, using fallback decomposition")
            return self._create_fallback_plan(strategies[0], top_tools, task_description)
        
        # Формируем промпт для планирования
        prompt = self._create_planning_prompt(task_description, strategies, algorithm_used, top_tools)
        
        try:
            # Вызов LLM
            response = self._llm.query_llm(
                question=prompt,
                temperature=0.2,
                max_tokens=2000
            )
            
            logger.info(f"ReWOO LLM response (first 300 chars): {response[:300]}")
            
            # Парсинг плана
            plan = self._parse_plan_from_response(response, strategies, top_tools, task_description)
            
            if plan and len(plan.steps) >= 2:
                for s in plan.steps:
                    logger.info(f"  ReWOO step: {s.step_id} '{s.description[:60]}' depends={s.depends_on}")
                return plan
            
            # Если LLM вернул 1-шаговый план, делаем fallback
            logger.warning("LLM returned single-step plan, using fallback decomposition")
            return self._create_fallback_plan(strategies[0], top_tools, task_description)
        
        except Exception as e:
            logger.error(f"Error creating ReWOO plan: {e}")
            return self._create_fallback_plan(strategies[0], top_tools, task_description)
    
    def _create_planning_prompt(
        self,
        task_description: str,
        strategies: List[List],
        algorithm: str,
        top_tools: Optional[List] = None
    ) -> str:
        """
        Создать промпт для планирования.
        
        Включает:
        - Описание задачи
        - Доступные инструменты (топ-10)
        - Найденные цепочки
        - Требование создать детальный план
        """
        prompt = f"""Создай JSON-план выполнения задачи.

ЗАДАЧА:
{task_description}

"""
        
        # Добавляем топ-10 инструментов
        if top_tools:
            prompt += "ДОСТУПНЫЕ ИНСТРУМЕНТЫ (топ по репутации):\n"
            for i, tool in enumerate(top_tools[:10], 1):
                name = tool.metadata.tool_name if hasattr(tool, 'metadata') else str(tool)
                rep = tool.metadata.reputation if hasattr(tool, 'metadata') else 0.5
                tool_type = tool._agent.type if hasattr(tool, '_agent') else "unknown"
                inp = tool.input.format if hasattr(tool, 'input') else "?"
                out = tool.output.format if hasattr(tool, 'output') else "?"
                desc = tool.metadata.description if hasattr(tool, 'metadata') else ""
                
                prompt += f"  {i}. {name} (тип: {tool_type}, репутация: {rep:.0%})\n"
                prompt += f"     Вход: {inp} → Выход: {out}\n"
                if desc:
                    prompt += f"     Описание: {desc}\n"
            prompt += "\n"
        
        # Добавляем цепочки
        if strategies:
            prompt += f"НАЙДЕННЫЕ ЦЕПОЧКИ (от алгоритма {algorithm}):\n"
            for i, strategy in enumerate(strategies[:5], 1):
                tool_names = [
                    t.metadata.tool_name if hasattr(t, 'metadata') else str(t)
                    for t in strategy
                ]
                chain_str = " → ".join(tool_names)
                prompt += f"  Цепочка {i}: {chain_str}\n"
            prompt += "\n"
        
        prompt += f"""ЗАДАНИЕ:
Проанализируй задачу "{task_description}" и создай план выполнения.

КРИТИЧЕСКИ ВАЖНО:
1. Если в задаче НЕСКОЛЬКО тем/вопросов (через "и", "а также", ",") — ОБЯЗАТЕЛЬНО разбей на отдельные шаги.
2. Каждая независимая подзадача: depends_on = []. Она должна получить КОНКРЕТНУЮ часть общего запроса.
3. Последний шаг: ОБЪЕДИНЯЕТ результаты всех предыдущих шагов (depends_on = ["step-1", "step-2", ...]).
4. description = КОНКРЕТНАЯ подзадача (НЕ "генерация ответа", а например "Объясни принцип работы самолета").

ПРИМЕРЫ ПРАВИЛЬНОЙ ДЕКОМПОЗИЦИИ:

Задача: "Объясни принцип работы самолета и подводной лодки"
{{
  "reasoning": "Задача содержит 2 независимые темы. Разбиваем на 2 подзадачи и финальное объединение.",
  "steps": [
    {{"step_id": "step-1", "tool_name": "General QA", "description": "Объясни принцип работы самолета", "depends_on": [], "expected_output": "Объяснение про самолет"}},
    {{"step_id": "step-2", "tool_name": "General QA", "description": "Объясни принцип работы подводной лодки", "depends_on": [], "expected_output": "Объяснение про лодку"}},
    {{"step_id": "step-3", "tool_name": "General QA", "description": "Объедини полученные объяснения про самолет и подводную лодку в один связный текст", "depends_on": ["step-1", "step-2"], "expected_output": "Объединённый ответ"}}
  ],
  "estimated_time": 6.0,
  "estimated_cost": 0.06
}}

Задача: "Сравни Python и Java"
{{
  "reasoning": "Задача требует сравнения 2 технологий. Разбиваем на анализ каждой + сравнение.",
  "steps": [
    {{"step_id": "step-1", "tool_name": "General QA", "description": "Опиши преимущества и недостатки Python", "depends_on": [], "expected_output": "Анализ Python"}},
    {{"step_id": "step-2", "tool_name": "General QA", "description": "Опиши преимущества и недостатки Java", "depends_on": [], "expected_output": "Анализ Java"}},
    {{"step_id": "step-3", "tool_name": "General QA", "description": "Составь сравнение Python и Java на основе анализов", "depends_on": ["step-1", "step-2"], "expected_output": "Сравнительная таблица"}}
  ],
  "estimated_time": 6.0,
  "estimated_cost": 0.06
}}

Задача: "Напиши рассказ про кота"
{{
  "reasoning": "Задача простая, одна тема. Один шаг достаточен.",
  "steps": [
    {{"step_id": "step-1", "tool_name": "Creative Responder", "description": "Напиши рассказ про кота", "depends_on": [], "expected_output": "Рассказ"}}
  ],
  "estimated_time": 3.0,
  "estimated_cost": 0.03
}}

ТЕПЕРЬ ТВОЯ ЗАДАЧА:
Задача: "{task_description}"

Выведи ТОЛЬКО JSON (без пояснений):
"""
        
        return prompt
    
    def _parse_plan_from_response(
        self,
        response: str,
        original_strategies: List[List],
        top_tools: Optional[List] = None,
        task_description: str = ""
    ) -> Optional[ReWOOPlan]:
        """
        Распарсить план из ответа LLM.
        
        Args:
            response: Ответ от LLM
            original_strategies: Оригинальные цепочки (для fallback)
            
        Returns:
            ReWOOPlan или None
        """
        try:
            # Извлечение JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON in ReWOO plan response")
                return self._create_fallback_plan(original_strategies[0], top_tools, task_description)
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Создание шагов
            steps = []
            for step_data in data.get('steps', []):
                step = ReWOOStep(
                    step_id=step_data.get('step_id', f"step-{len(steps)}"),
                    tool_name=step_data.get('tool_name', 'Unknown'),
                    description=step_data.get('description', ''),
                    depends_on=step_data.get('depends_on', []),
                    expected_output=step_data.get('expected_output', '')
                )
                steps.append(step)
            
            plan = ReWOOPlan(
                steps=steps,
                reasoning=data.get('reasoning', ''),
                estimated_time=data.get('estimated_time', 5.0),
                estimated_cost=data.get('estimated_cost', 0.05)
            )
            
            return plan
        
        except Exception as e:
            logger.error(f"Error parsing ReWOO plan: {e}")
            
            # Fallback: создаем декомпозированный план
            return self._create_fallback_plan(original_strategies[0], top_tools, task_description)
    
    def _create_fallback_plan(
        self,
        strategy: List,
        top_tools: Optional[List] = None,
        task_description: str = ""
    ) -> ReWOOPlan:
        """
        Создать план из цепочки (fallback если LLM недоступен).
        
        Args:
            strategy: Цепочка инструментов
            top_tools: Топ инструментов для декомпозиции
            task_description: Оригинальный запрос пользователя
            
        Returns:
            ReWOOPlan с минимум 2 шагами
        """
        steps = []
        task = task_description or "Выполнить задачу"
        
        if len(strategy) == 1 and top_tools and len(top_tools) >= 2:
            main_tool_name = strategy[0].metadata.tool_name if hasattr(strategy[0], 'metadata') else str(strategy[0])
            
            is_multi_topic = False
            multi_keywords = [" и ", ", и ", " а также ", " плюс ", " vs ", " или "]
            for kw in multi_keywords:
                if kw in task.lower():
                    is_multi_topic = True
                    break
            
            # Также проверяем: есть ли несколько вопросительных слов
            question_words = ["как", "что", "почему", "где", "когда", "кто"]
            question_count = sum(1 for qw in question_words if qw in task.lower())
            if question_count >= 2:
                is_multi_topic = True
            
            generator = None
            merger = None
            
            for t in top_tools:
                t_type = t._agent.type if hasattr(t, '_agent') else ""
                t_name = t.metadata.tool_name if hasattr(t, 'metadata') else str(t)
                
                if not generator and t_type in ("qa", "universal", "content_generation", "writing"):
                    generator = t_name
                elif not merger and t_type in ("qa", "universal", "content_generation"):
                    merger = t_name
            
            if not generator:
                generator = main_tool_name
            if not merger:
                merger = generator  # Тот же инструмент для объединения
            
            if is_multi_topic:
                # Многотемный вопрос: пробуем разбить
                # Простое разбиение по союзам
                parts = []
                for sep in [" и ", ", и ", " а также "]:
                    if sep in task:
                        parts = [p.strip() for p in task.split(sep, 1)]
                        break
                
                if len(parts) == 2:
                    steps = [
                        ReWOOStep(
                            step_id="step-0",
                            tool_name=generator,
                            description=parts[0],
                            depends_on=[],
                            expected_output=f"Ответ на: {parts[0][:50]}"
                        ),
                        ReWOOStep(
                            step_id="step-1",
                            tool_name=generator,
                            description=parts[1],
                            depends_on=[],
                            expected_output=f"Ответ на: {parts[1][:50]}"
                        ),
                        ReWOOStep(
                            step_id="step-2",
                            tool_name=merger,
                            description=f"Объедини ответы на вопросы: {parts[0][:40]} и {parts[1][:40]}",
                            depends_on=["step-0", "step-1"],
                            expected_output="Объединённый ответ"
                        ),
                    ]
                    
                    return ReWOOPlan(
                        steps=steps,
                        reasoning=f"Fallback: многотемная декомпозиция (2 подвопроса + объединение)",
                        estimated_time=6.0,
                        estimated_cost=0.06
                    )
            
            # Обычный вопрос: генерация без проверки (проверка часто портит ответ)
            steps = [
                ReWOOStep(
                    step_id="step-0",
                    tool_name=generator,
                    description=task,
                    depends_on=[],
                    expected_output="Ответ"
                ),
            ]
            
            return ReWOOPlan(
                steps=steps,
                reasoning=f"Fallback: прямой ответ через {generator}",
                estimated_time=3.0,
                estimated_cost=0.03
            )
        
        # Стандартный fallback — оборачиваем существующую цепочку
        for i, tool in enumerate(strategy):
            tool_name = tool.metadata.tool_name if hasattr(tool, 'metadata') else str(tool)
            
            step = ReWOOStep(
                step_id=f"step-{i}",
                tool_name=tool_name,
                description=f"Выполнение: {tool_name}",
                depends_on=[f"step-{i-1}"] if i > 0 else [],
                expected_output="Результат"
            )
            steps.append(step)
        
        return ReWOOPlan(
            steps=steps,
            reasoning="Fallback plan на основе графовой стратегии",
            estimated_time=len(steps) * 2.0,
            estimated_cost=len(steps) * 0.02
        )
