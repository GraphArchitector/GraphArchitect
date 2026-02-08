import logging
import json
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class LLMCriticScore:
    """
    Оценка от LLM критика.
    
    Содержит общий балл качества, детализированные оценки по критериям,
    обоснование и метаданные для аудита и reproducibility.
    """
    
    overall_score: float
    correctness: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    
    reasoning: str = ""
    suggestions: str = ""
    
    critic_type: str = "llm"
    model_used: str = ""
    
    timestamp: str = ""
    prompt_version: str = "1.0"
    model_version: str = ""
    evaluation_id: str = ""
    confidence: float = 1.0
    raw_response: str = ""


class LLMCritic:
    """
    LLM судья для оценки качества ответов систем.
    
    Использует либо VLLM (локально) либо OpenRouter (API) для оценки
    качества ответов на задачи. Поддерживает детальную оценку по
    четырем критериям: правильность, полнота, релевантность и ясность.
    
    Атрибуты:
        backend: Тип бэкенда ('openrouter' или 'vllm')
        model_name: Название используемой модели
        temperature: Температура для генерации (низкая для consistency)
        detailed_evaluation: Использовать ли детальную оценку по критериям
    """
    
    PROMPT_VERSION = "2.0"
    
    def __init__(
        self,
        backend: str = "openrouter",
        model_name: str = "openai/gpt-4",
        vllm_host: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        temperature: float = 0.2,
        detailed_evaluation: bool = True,
        max_tokens: int = 500
    ):
        """
        Инициализация LLM критика.
        
        Аргументы:
            backend: Бэкенд для LLM ('openrouter' или 'vllm')
            model_name: Название модели
            vllm_host: URL VLLM сервера (для backend='vllm')
            openrouter_api_key: API ключ OpenRouter
            temperature: Температура генерации (0.0-1.0)
            detailed_evaluation: Детальная оценка по критериям
            max_tokens: Максимальное число токенов для ответа
            
        Вызывает:
            ValueError: Если указан неподдерживаемый бэкенд
        """
        self._backend = backend
        self._model_name = model_name
        self._temperature = temperature
        self._detailed_evaluation = detailed_evaluation
        self._max_tokens = max_tokens
        
        if backend == "openrouter":
            self._init_openrouter(openrouter_api_key)
        elif backend == "vllm":
            self._init_vllm(vllm_host)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Use 'openrouter' or 'vllm'"
            )
        
        logger.info(
            f"LLM Critic initialized: backend={backend}, "
            f"model={model_name}, detailed={detailed_evaluation}"
        )
    
    def _init_openrouter(self, api_key: Optional[str]):
        """
        Инициализация OpenRouter бэкенда.
        
        Аргументы:
            api_key: API ключ для OpenRouter
            
        Вызывает:
            ImportError: Если не удается загрузить OpenRouterLLM
        """
        try:
            from ...tools.ApiTools.OpenRouterTool.openrouter_llm import OpenRouterLLM
            
            self._llm = OpenRouterLLM(
                api_key=api_key,
                model_name=self._model_name,
                system_prompt=(
                    "You are an expert AI critic evaluating task completion "
                    "quality. Provide structured evaluations with clear reasoning."
                )
            )
            
            logger.info("OpenRouter backend initialized successfully")
        
        except ImportError as error:
            logger.error(f"Failed to import OpenRouterLLM: {error}")
            raise
    
    def _init_vllm(self, vllm_host: Optional[str]):
        """
        Инициализация VLLM бэкенда.
        
        Аргументы:
            vllm_host: URL хоста VLLM сервера
            
        Вызывает:
            ImportError: Если не удается загрузить VLLMApi
        """
        if not vllm_host:
            vllm_host = os.getenv("VLLM_HOST", "http://localhost:8000")
        
        try:
            from ...tools.ApiTools.VLLMTool.VLLMApi import VLLMApi
            
            self._llm = VLLMApi(
                vllm_host=vllm_host,
                model_name=self._model_name,
                system_prompt=(
                    "You are an expert AI critic evaluating task completion "
                    "quality. Provide structured evaluations with clear reasoning."
                )
            )
            
            logger.info(f"VLLM backend initialized: {vllm_host}")
        
        except ImportError as error:
            logger.error(f"Failed to import VLLMApi: {error}")
            raise
    
    def evaluate_answer(
        self,
        task: str,
        answer: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> LLMCriticScore:
        """
        Оценить качество ответа на задачу.
        
        Аргументы:
            task: Исходная задача (текст)
            answer: Ответ системы
            context: Дополнительный контекст (опционально)
                - execution_time: время выполнения (секунды)
                - tools_used: список используемых инструментов
                - cost: стоимость выполнения
            task_id: Идентификатор задачи для логирования
            
        Возвращает:
            LLMCriticScore с оценками и метаданными
        """
        evaluation_id = str(uuid.uuid4())
        
        if task_id:
            logger.info(
                f"Starting evaluation: task_id={task_id}, "
                f"evaluation_id={evaluation_id}"
            )
        
        evaluation_prompt = self._create_evaluation_prompt(task, answer, context)
        
        try:
            response = self._call_llm(evaluation_prompt)
            
            score = self._parse_evaluation_response(response)
            
            if score:
                score.model_used = self._model_name
                score.evaluation_id = evaluation_id
                score.timestamp = datetime.utcnow().isoformat()
                score.prompt_version = self.PROMPT_VERSION
                score.raw_response = response[:200] if len(response) > 200 else response
                
                logger.info(
                    f"Evaluation completed: task_id={task_id}, "
                    f"overall={score.overall_score:.3f}, "
                    f"correctness={score.correctness:.3f}, "
                    f"completeness={score.completeness:.3f}, "
                    f"relevance={score.relevance:.3f}, "
                    f"clarity={score.clarity:.3f}"
                )
            else:
                logger.warning(
                    f"Failed to parse evaluation response for task_id={task_id}"
                )
                score = self._create_fallback_score(evaluation_id)
            
            return score
        
        except Exception as error:
            logger.error(
                f"Error during evaluation for task_id={task_id}: {error}"
            )
            return self._create_fallback_score(evaluation_id)
    
    def _create_fallback_score(self, evaluation_id: str) -> LLMCriticScore:
        """
        Создать fallback оценку при ошибке.
        
        Аргументы:
            evaluation_id: Идентификатор оценки
            
        Возвращает:
            LLMCriticScore с нейтральными значениями
        """
        return LLMCriticScore(
            overall_score=0.5,
            correctness=0.5,
            completeness=0.5,
            relevance=0.5,
            clarity=0.5,
            reasoning="Error during evaluation. Using default score.",
            model_used=self._model_name,
            evaluation_id=evaluation_id,
            timestamp=datetime.utcnow().isoformat(),
            confidence=0.0
        )
    
    def _create_evaluation_prompt(
        self,
        task: str,
        answer: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Создать промпт для оценки качества ответа.
        
        Формирует структурированный промпт с критериями оценки и
        требованиями к формату ответа.
        
        Аргументы:
            task: Текст задачи
            answer: Текст ответа
            context: Дополнительный контекст выполнения
            
        Возвращает:
            Промпт для LLM
        """
        context = context or {}
        
        prompt = (
            "You are an expert evaluator assessing AI system responses. "
            "Evaluate the following task and answer with precision and fairness.\n\n"
        )
        
        prompt += f"TASK:\n{task}\n\n"
        prompt += f"ANSWER:\n{answer}\n"
        
        if context:
            prompt += "\nEXECUTION CONTEXT:\n"
            
            if 'execution_time' in context:
                prompt += f"- Execution time: {context['execution_time']:.3f}s\n"
            
            if 'tools_used' in context:
                tools_str = ", ".join(context['tools_used'])
                prompt += f"- Tools used: {tools_str}\n"
            
            if 'cost' in context:
                prompt += f"- Cost: ${context['cost']:.6f}\n"
            
            if 'iterations' in context:
                prompt += f"- Iterations: {context['iterations']}\n"
        
        if self._detailed_evaluation:
            prompt += self._create_detailed_criteria()
        else:
            prompt += self._create_simple_criteria()
        
        return prompt
    
    def _create_detailed_criteria(self) -> str:
        """
        Создать детальные критерии оценки.
        
        Возвращает:
            Текст критериев и требований к формату ответа
        """
        return (
            "\n\nEVALUATION CRITERIA (score each 0-10):\n\n"
            
            "1. CORRECTNESS (Correctness):\n"
            "   - Does the answer correctly solve the task?\n"
            "   - Are there any factual errors or contradictions?\n"
            "   - Is the logic sound and consistent?\n"
            "   Score definition:\n"
            "   0-3: Major errors, incorrect solution\n"
            "   4-6: Partially correct, some issues\n"
            "   7-8: Mostly correct with minor issues\n"
            "   9-10: Completely correct, no errors\n\n"
            
            "2. COMPLETENESS (Completeness):\n"
            "   - Does the answer fully address all aspects of the task?\n"
            "   - Are all required elements covered?\n"
            "   - Are there any significant omissions?\n"
            "   Score definition:\n"
            "   0-3: Major omissions, incomplete\n"
            "   4-6: Partially complete, some gaps\n"
            "   7-8: Mostly complete with minor gaps\n"
            "   9-10: Fully comprehensive\n\n"
            
            "3. RELEVANCE (Relevance):\n"
            "   - Is the answer relevant to the task?\n"
            "   - Are there unnecessary or off-topic elements?\n"
            "   - Does it stay focused on what matters?\n"
            "   Score definition:\n"
            "   0-3: Many irrelevant elements\n"
            "   4-6: Some irrelevant content mixed in\n"
            "   7-8: Mostly relevant with minor distractions\n"
            "   9-10: Perfectly focused and relevant\n\n"
            
            "4. CLARITY (Clarity):\n"
            "   - Is the answer easy to understand?\n"
            "   - Is it well-structured and organized?\n"
            "   - Would a reader easily follow the logic?\n"
            "   Score definition:\n"
            "   0-3: Confusing, poorly structured\n"
            "   4-6: Somewhat unclear, needs improvement\n"
            "   7-8: Clear and well-organized\n"
            "   9-10: Excellent clarity and structure\n\n"
            
            "RESPONSE FORMAT (mandatory JSON):\n"
            "Return ONLY valid JSON with no additional text:\n"
            "{\n"
            "  \"correctness\": <integer 0-10>,\n"
            "  \"completeness\": <integer 0-10>,\n"
            "  \"relevance\": <integer 0-10>,\n"
            "  \"clarity\": <integer 0-10>,\n"
            "  \"overall\": <integer 0-10>,\n"
            "  \"reasoning\": \"Brief explanation of your scores\",\n"
            "  \"suggestions\": \"Specific suggestions for improvement (if any)\"\n"
            "}"
        )
    
    def _create_simple_criteria(self) -> str:
        """
        Создать простые критерии оценки.
        
        Возвращает:
            Текст критериев и требований к формату ответа
        """
        return (
            "\n\nEVALUATION:\n"
            "Rate the overall quality of the answer (0-10 scale):\n"
            "0-3: Poor quality, major issues\n"
            "4-6: Average quality, needs improvement\n"
            "7-8: Good quality, minor issues\n"
            "9-10: Excellent quality\n\n"
            
            "RESPONSE FORMAT (mandatory JSON):\n"
            "Return ONLY valid JSON with no additional text:\n"
            "{\n"
            "  \"overall\": <integer 0-10>,\n"
            "  \"reasoning\": \"Brief explanation of your score\"\n"
            "}"
        )
    
    def _call_llm(self, prompt: str) -> str:
        """
        Вызвать LLM для получения оценки.
        
        Аргументы:
            prompt: Промпт для оценки
            
        Возвращает:
            Ответ от LLM
            
        Вызывает:
            Exception: Если вызов LLM завершился с ошибкой
        """
        logger.debug(
            f"LLM Call: model={self._model_name}, "
            f"temperature={self._temperature}, "
            f"max_tokens={self._max_tokens}"
        )
        
        try:
            if self._backend == "openrouter":
                response = self._llm.query_llm(
                    question=prompt,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens
                )
            elif self._backend == "vllm":
                response = self._llm.query_llm(
                    question=prompt,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens
                )
            else:
                raise ValueError(f"Unknown backend: {self._backend}")
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            return response
        
        except Exception as error:
            logger.error(f"LLM call failed: {error}")
            raise
    
    def _parse_evaluation_response(self, response: str) -> Optional[LLMCriticScore]:
        """
        Распарсить ответ LLM и извлечь оценки.
        
        Аргументы:
            response: Ответ от LLM
            
        Возвращает:
            LLMCriticScore или None при ошибке парсинга
        """
        try:
            json_str = self._extract_json(response)
            
            if not json_str:
                logger.error("No valid JSON found in LLM response")
                logger.debug(f"Response: {response[:300]}")
                return None
            
            data = json.loads(json_str)
            
            return self._create_score_from_data(data)
        
        except json.JSONDecodeError as error:
            logger.error(f"JSON parsing failed: {error}")
            logger.debug(f"Response: {response[:300]}")
            return None
        
        except Exception as error:
            logger.error(f"Error parsing evaluation: {error}")
            return None
    
    def _extract_json(self, response: str) -> Optional[str]:
        """
        Извлечь JSON из ответа LLM.
        
        Поддерживает JSON в markdown блоках и в чистом виде.
        
        Аргументы:
            response: Ответ LLM
            
        Возвращает:
            JSON строка или None
        """
        import re
        
        # Попытка 1: JSON в markdown блоке
        markdown_match = re.search(
            r'```(?:json)?\s*(\{[^`]*?\})\s*```',
            response,
            re.DOTALL
        )
        if markdown_match:
            return markdown_match.group(1).strip()
        
        # Попытка 2: Просто JSON объект
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]
        
        return None
    
    def _create_score_from_data(self, data: Dict[str, Any]) -> LLMCriticScore:
        """
        Создать LLMCriticScore из распарсенных данных.
        
        Аргументы:
            data: Распарсенные данные из JSON
            
        Возвращает:
            LLMCriticScore с валидированными значениями
        """
        def normalize_score(value, default: float = 5.0) -> float:
            """Нормализовать оценку в диапазон 0-1."""
            if isinstance(value, (int, float)):
                score = float(value)
                if 0 <= score <= 10:
                    return score / 10.0
                elif 0 <= score <= 1:
                    return score
            return default / 10.0
        
        if self._detailed_evaluation:
            correctness = normalize_score(data.get('correctness'))
            completeness = normalize_score(data.get('completeness'))
            relevance = normalize_score(data.get('relevance'))
            clarity = normalize_score(data.get('clarity'))
            overall = normalize_score(data.get('overall'))
        else:
            overall = normalize_score(data.get('overall'))
            correctness = overall
            completeness = overall
            relevance = overall
            clarity = overall
        
        score = LLMCriticScore(
            overall_score=overall,
            correctness=correctness,
            completeness=completeness,
            relevance=relevance,
            clarity=clarity,
            reasoning=str(data.get('reasoning', '')),
            suggestions=str(data.get('suggestions', '')),
            critic_type="llm"
        )
        
        return score
    
    def batch_evaluate(
        self,
        tasks_and_answers: List[Tuple[str, str, Optional[Dict]]],
        save_results: bool = False,
        output_file: Optional[str] = None
    ) -> List[LLMCriticScore]:
        """
        Batch оценка нескольких пар задача-ответ.
        
        Аргументы:
            tasks_and_answers: Список кортежей (task, answer, context)
            save_results: Сохранить ли результаты в файл
            output_file: Путь для сохранения результатов
            
        Возвращает:
            Список LLMCriticScore
        """
        scores = []
        
        logger.info(
            f"Starting batch evaluation: {len(tasks_and_answers)} items"
        )
        
        for idx, item in enumerate(tasks_and_answers, 1):
            if len(item) == 2:
                task, answer = item
                context = None
            elif len(item) == 3:
                task, answer, context = item
            else:
                logger.warning(f"Invalid item format at index {idx}")
                continue
            
            score = self.evaluate_answer(
                task,
                answer,
                context,
                task_id=f"batch_{idx}"
            )
            scores.append(score)
        
        logger.info(f"Batch evaluation completed: {len(scores)} items")
        
        if save_results and output_file:
            self._save_results(scores, output_file)
        
        return scores
    
    def _save_results(self, scores: List[LLMCriticScore], output_file: str):
        """
        Сохранить результаты оценок в JSON файл.
        
        Аргументы:
            scores: Список оценок
            output_file: Путь файла для сохранения
        """
        try:
            results = []
            for score in scores:
                results.append({
                    'evaluation_id': score.evaluation_id,
                    'timestamp': score.timestamp,
                    'overall': round(score.overall_score, 4),
                    'correctness': round(score.correctness, 4),
                    'completeness': round(score.completeness, 4),
                    'relevance': round(score.relevance, 4),
                    'clarity': round(score.clarity, 4),
                    'reasoning': score.reasoning,
                    'suggestions': score.suggestions,
                    'model': score.model_used,
                    'prompt_version': score.prompt_version,
                    'confidence': score.confidence
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
        
        except Exception as error:
            logger.error(f"Failed to save results: {error}")
    
    def evaluate_with_consensus(
        self,
        task: str,
        answer: str,
        num_evaluators: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMCriticScore:
        """
        Получить оценку от нескольких независимых судей и усреднить.
        
        Обеспечивает лучшую reproducibility и надежность оценок.
        
        Аргументы:
            task: Текст задачи
            answer: Текст ответа
            num_evaluators: Количество судей
            context: Дополнительный контекст
            
        Возвращает:
            Усредненная LLMCriticScore
        """
        logger.info(
            f"Starting consensus evaluation with {num_evaluators} judges"
        )
        
        individual_scores = []
        
        for judge_id in range(num_evaluators):
            score = self.evaluate_answer(
                task,
                answer,
                context,
                task_id=f"consensus_judge_{judge_id + 1}"
            )
            individual_scores.append(score)
        
        # Усреднение оценок
        avg_score = LLMCriticScore(
            overall_score=statistics.mean([s.overall_score for s in individual_scores]),
            correctness=statistics.mean([s.correctness for s in individual_scores]),
            completeness=statistics.mean([s.completeness for s in individual_scores]),
            relevance=statistics.mean([s.relevance for s in individual_scores]),
            clarity=statistics.mean([s.clarity for s in individual_scores]),
            reasoning=f"Consensus from {num_evaluators} independent judges",
            suggestions="",
            model_used=self._model_name,
            evaluation_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(
            f"Consensus evaluation complete: "
            f"overall={avg_score.overall_score:.4f}"
        )
        
        return avg_score
    
    def get_statistics(self, scores: List[LLMCriticScore]) -> Dict[str, Any]:
        """
        Вычислить статистику по набору оценок.
        
        Аргументы:
            scores: Список LLMCriticScore
            
        Возвращает:
            Словарь со статистическими показателями
        """
        if not scores:
            return {}
        
        overall_scores = [s.overall_score for s in scores]
        correctness_scores = [s.correctness for s in scores]
        completeness_scores = [s.completeness for s in scores]
        relevance_scores = [s.relevance for s in scores]
        clarity_scores = [s.clarity for s in scores]
        
        def calc_stats(values):
            return {
                'mean': statistics.mean(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values)
            }
        
        stats = {
            'count': len(scores),
            'overall': calc_stats(overall_scores),
            'correctness': calc_stats(correctness_scores),
            'completeness': calc_stats(completeness_scores),
            'relevance': calc_stats(relevance_scores),
            'clarity': calc_stats(clarity_scores),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(
            f"Statistics computed for {len(scores)} scores. "
            f"Mean overall: {stats['overall']['mean']:.4f}"
        )
        
        return stats
    
    def compare_systems(
        self,
        system_a_results: List[LLMCriticScore],
        system_b_results: List[LLMCriticScore],
        system_a_name: str = "System A",
        system_b_name: str = "System B"
    ) -> Dict[str, Any]:
        """
        Сравнить результаты двух систем.
        
        Аргументы:
            system_a_results: Оценки системы A
            system_b_results: Оценки системы B
            system_a_name: Название системы A
            system_b_name: Название системы B
            
        Возвращает:
            Словарь с результатами сравнения
        """
        stats_a = self.get_statistics(system_a_results)
        stats_b = self.get_statistics(system_b_results)
        
        def calc_improvement(value_a, value_b):
            """Вычислить улучшение в процентах."""
            if value_b == 0:
                return 0.0
            return ((value_a - value_b) / value_b) * 100
        
        comparison = {
            'system_a': {
                'name': system_a_name,
                'count': stats_a.get('count', 0),
                'overall_mean': stats_a.get('overall', {}).get('mean', 0),
                'correctness_mean': stats_a.get('correctness', {}).get('mean', 0),
                'completeness_mean': stats_a.get('completeness', {}).get('mean', 0),
                'relevance_mean': stats_a.get('relevance', {}).get('mean', 0),
                'clarity_mean': stats_a.get('clarity', {}).get('mean', 0),
            },
            'system_b': {
                'name': system_b_name,
                'count': stats_b.get('count', 0),
                'overall_mean': stats_b.get('overall', {}).get('mean', 0),
                'correctness_mean': stats_b.get('correctness', {}).get('mean', 0),
                'completeness_mean': stats_b.get('completeness', {}).get('mean', 0),
                'relevance_mean': stats_b.get('relevance', {}).get('mean', 0),
                'clarity_mean': stats_b.get('clarity', {}).get('mean', 0),
            },
            'improvement_percent': {
                'overall': calc_improvement(
                    stats_a.get('overall', {}).get('mean', 0),
                    stats_b.get('overall', {}).get('mean', 1)
                ),
                'correctness': calc_improvement(
                    stats_a.get('correctness', {}).get('mean', 0),
                    stats_b.get('correctness', {}).get('mean', 1)
                ),
                'completeness': calc_improvement(
                    stats_a.get('completeness', {}).get('mean', 0),
                    stats_b.get('completeness', {}).get('mean', 1)
                ),
                'relevance': calc_improvement(
                    stats_a.get('relevance', {}).get('mean', 0),
                    stats_b.get('relevance', {}).get('mean', 1)
                ),
                'clarity': calc_improvement(
                    stats_a.get('clarity', {}).get('mean', 0),
                    stats_b.get('clarity', {}).get('mean', 1)
                ),
            }
        }
        
        logger.info(
            f"Comparison: {system_a_name} vs {system_b_name}. "
            f"Overall improvement: {comparison['improvement_percent']['overall']:.2f}%"
        )
        
        return comparison