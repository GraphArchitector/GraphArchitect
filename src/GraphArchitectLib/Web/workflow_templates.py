"""
Шаблоны Workflow с шагами и конкурентным выбором агентов
"""
from models import WorkflowChain, WorkflowStep, SelectionCriteria


# Шаблоны workflow
WORKFLOW_TEMPLATES = {
    "customer_support": WorkflowChain(
        chat_id="template_customer_support",
        name="Customer Support Resolution",
        description="Обработка запроса клиента с выбором лучшего агента на каждом шаге",
        request_type="text",
        steps=[
            WorkflowStep(
                id="step-1",
                name="Classify Request",
                order=1,
                description="Определить тип и приоритет запроса",
                status="pending",
                candidateAgents=[
                    "agent-classifier-gpt4",
                    "agent-classifier-claude",
                    "agent-classifier-local",
                    "agent-classifier-fast"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="best_quality_score",
                    timeout=8000
                )
            ),
            WorkflowStep(
                id="step-2",
                name="Generate Response",
                order=2,
                description="Сгенерировать ответ клиенту",
                status="pending",
                candidateAgents=[
                    "agent-responder-creative",
                    "agent-responder-formal",
                    "agent-responder-technical",
                    "agent-responder-friendly"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="best_quality_score",
                    timeout=12000
                )
            ),
            WorkflowStep(
                id="step-3",
                name="Quality Check",
                order=3,
                description="Проверка качества ответа",
                status="pending",
                candidateAgents=[
                    "agent-qa-strict",
                    "agent-qa-balanced",
                    "agent-qa-fast"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="fastest_response",
                    timeout=5000
                )
            )
        ]
    ),
    
    "code_review": WorkflowChain(
        chat_id="template_code_review",
        name="Code Review Pipeline",
        description="Автоматический код-ревью с параллельным выбором анализаторов",
        request_type="text",
        steps=[
            WorkflowStep(
                id="step-1",
                name="Parse Code",
                order=1,
                description="Парсинг и анализ структуры кода",
                status="pending",
                candidateAgents=[
                    "agent-parser-fast",
                    "agent-parser-deep",
                    "agent-parser-incremental"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="fastest_response",
                    timeout=6000
                )
            ),
            WorkflowStep(
                id="step-2",
                name="Find Issues",
                order=2,
                description="Поиск проблем безопасности и производительности",
                status="pending",
                candidateAgents=[
                    "agent-security-scanner",
                    "agent-performance-analyzer",
                    "agent-style-checker",
                    "agent-bug-detector",
                    "agent-complexity-analyzer"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="consensus",
                    timeout=15000
                )
            ),
            WorkflowStep(
                id="step-3",
                name="Generate Report",
                order=3,
                description="Создание отчета о результатах",
                status="pending",
                candidateAgents=[
                    "agent-reporter-detailed",
                    "agent-reporter-summary",
                    "agent-reporter-interactive"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="best_quality_score",
                    timeout=7000
                )
            )
        ]
    ),
    
    "content_creation": WorkflowChain(
        chat_id="template_content_creation",
        name="Content Creation Workflow",
        description="Создание контента от исследования до публикации",
        request_type="text",
        steps=[
            WorkflowStep(
                id="step-1",
                name="Research Topic",
                order=1,
                description="Исследование темы и сбор информации",
                status="pending",
                candidateAgents=[
                    "agent-web-scraper",
                    "agent-academic-searcher",
                    "agent-trend-analyzer"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="best_quality_score",
                    timeout=10000
                )
            ),
            WorkflowStep(
                id="step-2",
                name="Create Outline",
                order=2,
                description="Создание структуры контента",
                status="pending",
                candidateAgents=[
                    "agent-outliner-structured",
                    "agent-outliner-creative",
                    "agent-outliner-seo"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="best_quality_score",
                    timeout=8000
                )
            ),
            WorkflowStep(
                id="step-3",
                name="Write Content",
                order=3,
                description="Написание основного контента",
                status="pending",
                candidateAgents=[
                    "agent-writer-formal",
                    "agent-writer-casual",
                    "agent-writer-technical",
                    "agent-writer-storytelling"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="best_quality_score",
                    timeout=20000
                )
            ),
            WorkflowStep(
                id="step-4",
                name="Proofread & Polish",
                order=4,
                description="Корректура и финальная обработка",
                status="pending",
                candidateAgents=[
                    "agent-grammar-checker",
                    "agent-style-improver",
                    "agent-fact-checker"
                ],
                selectionCriteria=SelectionCriteria(
                    strategy="consensus",
                    timeout=9000
                )
            )
        ]
    )
}


def get_workflow_template(template_name: str) -> WorkflowChain:
    """Получить шаблон workflow"""
    # Если это один из новых алгоритмов планирования
    if template_name in ["dijkstra", "astar", "yen_3", "yen_5", "yen_10", "ant_3", "ant_5", "ant_10"]:
        # Возвращаем универсальный шаблон, который будет дополнен алгоритмом планирования
        names = {
            "dijkstra": "Dijkstra Planning",
            "astar": "A-star Planning",
            "yen_3": "Yen's Algorithm (Top-3)",
            "yen_5": "Yen's Algorithm (Top-5)",
            "yen_10": "Yen's Algorithm (Top-10)",
            "ant_3": "Ant Colony (Top-3)",
            "ant_5": "Ant Colony (Top-5)",
            "ant_10": "Ant Colony (Top-10)"
        }
        return WorkflowChain(
            chat_id="dynamic_workflow",
            name=names.get(template_name, "Custom Planning"),
            description=f"Генерация графа с использованием алгоритма {names.get(template_name)}",
            request_type="text",
            steps=WORKFLOW_TEMPLATES["customer_support"].steps # Используем шаги поддержки как базу
        )

    template = WORKFLOW_TEMPLATES.get(template_name)
    if template:
        # Возвращаем копию чтобы не изменять оригинал
        import copy
        return copy.deepcopy(template)
    return None


def get_all_templates() -> dict:
    """Получить все доступные шаблоны"""
    templates = {
        name: {
            "name": wf.name,
            "description": wf.description,
            "steps_count": len(wf.steps)
        }
        for name, wf in WORKFLOW_TEMPLATES.items()
    }
    
    # Добавляем новые алгоритмы
    planning_algs = {
        "dijkstra": "Dijkstra",
        "astar": "A-star",
        "yen_3": "Yen's Algorithm (Top-3)",
        "yen_5": "Yen's Algorithm (Top-5)",
        "yen_10": "Yen's Algorithm (Top-10)",
        "ant_3": "Ant Colony (Top-3)",
        "ant_5": "Ant Colony (Top-5)",
        "ant_10": "Ant Colony (Top-10)"
    }
    
    for alg_id, alg_name in planning_algs.items():
        templates[alg_id] = {
            "name": alg_name,
            "description": f"Графовое планирование: {alg_name}",
            "steps_count": 3
        }
        
    return templates