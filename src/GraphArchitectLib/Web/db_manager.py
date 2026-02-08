#!/usr/bin/env python
"""
Утилита управления SQLite базой данных.

Команды:
    python db_manager.py init           - Создать таблицы
    python db_manager.py load_agents    - Загрузить агентов из agent_library
    python db_manager.py list_agents    - Показать всех агентов
    python db_manager.py stats          - Статистика БД
    python db_manager.py clear          - Очистить все данные
    python db_manager.py backup         - Создать backup
"""

import sys
import argparse
from pathlib import Path
import shutil
from datetime import datetime

from database import get_database
from sqlite_repository import get_sqlite_repository


def init_database(args):
    """Initialize database."""
    print("\n" + "="*70)
    print("DATABASE INITIALIZATION")
    print("="*70)
    
    db = get_database(args.db_path)
    
    print(f"\n[OK] Database initialized: {args.db_path}")
    print("\nTables:")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table['name']}")
            count = cursor.fetchone()[0]
            print(f"  [OK] {table['name']:20} ({count} records)")
    
    print("\n" + "="*70)


def load_agents(args):
    """Load tools/agents from default library to database."""
    print("\n" + "="*70)
    print("LOADING TOOLS TO DATABASE")
    print("="*70)
    
    db = get_database(args.db_path)
    
    # Clear existing agents if force flag is set
    if args.force:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM agents")
            print("  [WARNING] Existing tools deleted")
    
    # Load agents
    db.insert_default_agents()
    
    # Show result
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM agents")
        count = cursor.fetchone()[0]
        
        print(f"\n[OK] Database now has {count} tools")
    
    print("="*70)


def list_agents(args):
    """Show list of all tools/agents."""
    print("\n" + "="*70)
    print("TOOL LIST FROM DATABASE")
    print("="*70)
    
    repo = get_sqlite_repository(args.db_path)
    agents = repo.get_all_agents()
    
    if not agents:
        print("\n  [WARNING] No tools found")
        print("  Run: python db_manager.py load_agents")
        return
    
    print(f"\nTotal tools: {len(agents)}\n")
    
    # Группируем по типу
    by_type = {}
    for agent in agents:
        if agent.type not in by_type:
            by_type[agent.type] = []
        by_type[agent.type].append(agent)
    
    for agent_type, agents_list in sorted(by_type.items()):
        print(f"\n{agent_type.upper()} ({len(agents_list)}):")
        print("-" * 70)
        
        for agent in sorted(agents_list, key=lambda a: a.name):
            metrics_str = ""
            if agent.metrics:
                score = agent.metrics.get('avgScore', 0)
                time_ms = agent.metrics.get('avgResponseTime', 0)
                metrics_str = f"score={score:.2f}, time={time_ms}ms"
            
            print(f"  {agent.icon} {agent.name:30} ${agent.cost:.3f} {metrics_str}")
    
    print("\n" + "="*70)


def show_statistics(args):
    """Show database statistics."""
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    
    db = get_database(args.db_path)
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Статистика по таблицам
        tables = [
            'agents', 'workflows', 'chats', 'documents', 
            'executions', 'feedbacks', 'tool_metrics'
        ]
        
        print("\nРазмер таблиц:")
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {table:20} {count:>6} записей")
            except:
                print(f"  {table:20} [не существует]")
        
        # Статистика выполнений
        cursor.execute("SELECT COUNT(*) FROM executions WHERE status = 'COMPLETED'")
        completed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM executions WHERE status = 'FAILED'")
        failed = cursor.fetchone()[0]
        
        total_exec = completed + failed
        
        if total_exec > 0:
            print(f"\nВыполнения:")
            print(f"  Завершено:  {completed} ({completed/total_exec*100:.1f}%)")
            print(f"  Провалено:  {failed} ({failed/total_exec*100:.1f}%)")
        
        # Средняя оценка качества
        cursor.execute("SELECT AVG(quality_score) FROM feedbacks")
        avg_quality = cursor.fetchone()[0]
        
        if avg_quality:
            print(f"\nКачество:")
            print(f"  Средняя оценка: {avg_quality:.3f}")
        
        # Размер файла БД
        db_file = Path(args.db_path)
        if db_file.exists():
            size_mb = db_file.stat().st_size / (1024 * 1024)
            print(f"\nФайл БД:")
            print(f"  Путь: {db_file.absolute()}")
            print(f"  Размер: {size_mb:.2f} МБ")
    
    print("\n" + "="*70)


def clear_data(args):
    """Clear all data."""
    print("\n" + "="*70)
    print("WARNING: CLEARING ALL DATA")
    print("="*70)
    
    if not args.force:
        response = input("\nAre you sure? All data will be deleted! (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled")
            return
    
    db = get_database(args.db_path)
    db.clear_all_data()
    
    print("\n[OK] All data cleared")
    print("="*70)


def backup_database(args):
    """Create database backup."""
    print("\n" + "="*70)
    print("DATABASE BACKUP")
    print("="*70)
    
    db_file = Path(args.db_path)
    
    if not db_file.exists():
        print(f"\n[ERROR] Database file not found: {args.db_path}")
        return
    
    # Create backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = db_file.parent / f"{db_file.stem}_backup_{timestamp}{db_file.suffix}"
    
    # Copy file
    shutil.copy2(db_file, backup_file)
    
    size_mb = backup_file.stat().st_size / (1024 * 1024)
    
    print(f"\n[OK] Backup created:")
    print(f"  File: {backup_file}")
    print(f"  Size: {size_mb:.2f} MB")
    print("\n" + "="*70)


def add_agent(args):
    """Add new tool/agent manually."""
    print("\n" + "="*70)
    print("ADD NEW TOOL")
    print("="*70)
    
    from models import Agent
    from sqlite_repository import get_sqlite_repository
    import uuid
    
    # Interactive input
    agent_id = input("Tool ID: ") or f"agent-{uuid.uuid4().hex[:8]}"
    name = input("Name: ") or "New Tool"
    agent_type = input("Type (classification/writing/research etc.): ") or "general"
    icon = input("Icon: ") or "T"
    color = input("Color (hex): ") or "#6366f1"
    specialization = input("Specialization: ") or ""
    cost = float(input("Cost ($): ") or "0.01")
    
    agent = Agent(
        id=agent_id,
        name=name,
        type=agent_type,
        icon=icon,
        color=color,
        specialization=specialization,
        capabilities=[],
        cost=cost,
        metrics={
            "avgScore": 0.5,
            "avgResponseTime": 3000
        }
    )
    
    repo = get_sqlite_repository(args.db_path)
    repo.save_agent(agent)
    
    print(f"\n[OK] Tool added: {agent.name} ({agent.id})")
    print("="*70)


def export_agents(args):
    """Export tools/agents to JSON."""
    print("\n" + "="*70)
    print("EXPORT TOOLS")
    print("="*70)
    
    repo = get_sqlite_repository(args.db_path)
    agents = repo.get_all_agents()
    
    output_file = args.output or "agents_export.json"
    
    # Convert to JSON
    agents_data = [
        {
            'id': a.id,
            'name': a.name,
            'type': a.type,
            'icon': a.icon,
            'color': a.color,
            'specialization': a.specialization,
            'capabilities': a.capabilities,
            'cost': a.cost,
            'metrics': a.metrics
        }
        for a in agents
    ]
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(agents_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Exported {len(agents)} tools to {output_file}")
    print("="*70)


def import_agents(args):
    """Import tools/agents from JSON."""
    print("\n" + "="*70)
    print("IMPORT TOOLS")
    print("="*70)
    
    import json
    from models import Agent
    from sqlite_repository import get_sqlite_repository
    
    input_file = args.input
    
    if not Path(input_file).exists():
        print(f"\n[ERROR] File not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        agents_data = json.load(f)
    
    repo = get_sqlite_repository(args.db_path)
    
    imported = 0
    for data in agents_data:
        agent = Agent(**data)
        repo.save_agent(agent)
        imported += 1
    
    print(f"\n[OK] Imported {imported} tools")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Утилита управления SQLite БД для GraphArchitect",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--db-path',
        default='grapharchitect.db',
        help='Путь к файлу БД (по умолчанию: grapharchitect.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # init
    subparsers.add_parser('init', help='Инициализировать БД (создать таблицы)')
    
    # load_agents
    load_parser = subparsers.add_parser('load_agents', help='Загрузить агентов из agent_library')
    load_parser.add_argument('--force', action='store_true', help='Перезаписать существующих')
    
    # list_agents
    subparsers.add_parser('list_agents', help='Показать всех агентов')
    
    # stats
    subparsers.add_parser('stats', help='Показать статистику БД')
    
    # clear
    clear_parser = subparsers.add_parser('clear', help='Очистить все данные')
    clear_parser.add_argument('--force', action='store_true', help='Без подтверждения')
    
    # backup
    subparsers.add_parser('backup', help='Создать backup БД')
    
    # add_agent
    subparsers.add_parser('add_agent', help='Добавить нового агента (интерактивно)')
    
    # export
    export_parser = subparsers.add_parser('export', help='Экспортировать агентов в JSON')
    export_parser.add_argument('--output', help='Выходной файл', default='agents_export.json')
    
    # import
    import_parser = subparsers.add_parser('import_agents', help='Импортировать агентов из JSON')
    import_parser.add_argument('--input', required=True, help='Входной JSON файл')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Выполняем команду
    commands = {
        'init': init_database,
        'load_agents': load_agents,
        'list_agents': list_agents,
        'stats': show_statistics,
        'clear': clear_data,
        'backup': backup_database,
        'add_agent': add_agent,
        'export': export_agents,
        'import_agents': import_agents
    }
    
    command_func = commands.get(args.command)
    if command_func:
        try:
            command_func(args)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"[ERROR] Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
