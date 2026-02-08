"""
Скрипт проверки БД стандартов коннекторов.
Выводит структуру таблиц и содержимое для верификации.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "connectors_standards.db"

def check_database():
    print(f"Checking database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Список таблиц
    print("\n--- Tables ---")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        print(f"- {table[0]}")
        
        # Вывод схемы таблицы
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"    {col[1]} ({col[2]})")

    # 2. Содержимое справочников
    print("\n--- Subtypes (first 5) ---")
    cursor.execute("SELECT * FROM subtypes LIMIT 5")
    for row in cursor.fetchall():
        print(row)

    print("\n--- Semantic Categories (first 5) ---")
    cursor.execute("SELECT * FROM semantic_categories LIMIT 5")
    for row in cursor.fetchall():
        print(row)
        
    print("\n--- Tools ---")
    cursor.execute("""
        SELECT t.name, t.group_id, 
               s_in.subtype_name, sc_in.subtype_name,
               s_out.subtype_name, sc_out.subtype_name
        FROM Tools t
        JOIN input_connectors ic ON t.input_connector_id = ic.id
        JOIN output_connectors oc ON t.output_connector_id = oc.id
        JOIN subtypes s_in ON ic.subtype_id = s_in.id
        JOIN semantic_categories sc_in ON ic.semantic_category_id = sc_in.id
        JOIN subtypes s_out ON oc.subtype_id = s_out.id
        JOIN semantic_categories sc_out ON oc.semantic_category_id = sc_out.id
    """)
    for row in cursor.fetchall():
        print(f"Tool: {row[0]} (Group: {row[1]})")
        print(f"  Input:  {row[2]} | {row[3]}")
        print(f"  Output: {row[4]} | {row[5]}")

    conn.close()

if __name__ == "__main__":
    check_database()
