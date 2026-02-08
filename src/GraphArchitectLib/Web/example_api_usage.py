"""
Пример использования Graph Architect API
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api"


def example_1_simple_message():
    """Простой пример: отправка сообщения"""
    print("\n=== Пример 1: Простое сообщение ===")
    
    chat_id = f"demo_chat_{int(time.time())}"
    
    # Отправляем сообщение
    response = requests.post(
        f"{BASE_URL}/chat/{chat_id}/message",
        data={
            "message": "Привет! Расскажи о Python",
            "files": []
        }
    )
    
    result = response.json()
    print(f"Chat ID: {result['chat_id']}")
    print(f"Ответ: {result['response_data']}")
    print(f"Использовано агентов: {len(result['workflow_used'])}")
    print(f"Время обработки: {result['processing_time']}s")


def example_2_streaming():
    """Пример со стримингом"""
    print("\n=== Пример 2: Стриминг ответа ===")
    
    chat_id = f"stream_chat_{int(time.time())}"
    
    response = requests.post(
        f"{BASE_URL}/chat/{chat_id}/message/stream",
        data={
            "message": "Объясни что такое машинное обучение",
            "files": None
        },
        stream=True
    )
    
    print("Ответ: ", end="", flush=True)
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "agent_start":
                print(f"\n[{chunk['content']} начал работу]", end=" ", flush=True)
            elif chunk["type"] == "agent_complete":
                print(f"[Завершил]", end=" ", flush=True)
    print("\n")


def example_3_workflow():
    """Пример создания custom workflow"""
    print("\n=== Пример 3: Создание custom workflow ===")
    
    chat_id = f"custom_chat_{int(time.time())}"
    
    # Создаем workflow для работы с изображениями
    response = requests.post(
        f"{BASE_URL}/chat/{chat_id}/workflow",
        data={
            "request_type": "image",
            "user_message": "Проанализируй изображение",
            "files": ["image1.jpg", "image2.png"]
        }
    )
    
    workflow = response.json()
    print(f"Создана workflow для chat_id: {workflow['chat_id']}")
    print(f"Тип: {workflow['workflow']['request_type']}")
    print("Агенты:")
    for agent in workflow['workflow']['agents']:
        print(f"  {agent['icon']} {agent['name']} - {agent.get('description', '')}")
    
    # Теперь отправляем сообщение - автоматически использует созданную workflow
    print("\nОтправка сообщения с использованием workflow...")
    msg_response = requests.post(
        f"{BASE_URL}/chat/{chat_id}/message",
        data={"message": "Опиши что на изображениях"}
    )
    
    result = msg_response.json()
    print(f"Ответ: {result['response_data']}")


def example_4_document_upload():
    """Пример загрузки документа"""
    print("\n=== Пример 4: Загрузка документа ===")
    
    chat_id = f"doc_chat_{int(time.time())}"
    
    # Создаем тестовый файл
    test_content = b"Test document content for API example"
    
    # Загружаем документ
    response = requests.post(
        f"{BASE_URL}/chat/{chat_id}/document",
        files={"file": ("test_doc.txt", test_content, "text/plain")}
    )
    
    document = response.json()
    print(f"Документ загружен:")
    print(f"  ID: {document['document_id']}")
    print(f"  Имя: {document['filename']}")
    print(f"  Размер: {document['size']} байт")
    print(f"  Путь: {document['path']}")
    
    # Получаем список документов чата
    docs_response = requests.get(f"{BASE_URL}/chat/{chat_id}/documents")
    documents = docs_response.json()
    print(f"\nВсего документов в чате: {len(documents)}")


def example_5_chat_management():
    """Пример управления чатами"""
    print("\n=== Пример 5: Управление чатами ===")
    
    # Список всех чатов
    response = requests.get(f"{BASE_URL}/chats")
    chats = response.json()
    print(f"Всего чатов: {len(chats)}")
    
    if chats:
        # Информация о первом чате
        chat_id = chats[0]['chat_id']
        chat_response = requests.get(f"{BASE_URL}/chat/{chat_id}")
        chat_info = chat_response.json()
        
        print(f"\nИнформация о чате {chat_id}:")
        print(f"  Создан: {chat_info['created_at']}")
        print(f"  Последняя активность: {chat_info['last_activity']}")
        print(f"  Документов: {len(chat_info['documents'])}")
        if chat_info['workflow_chain']:
            print(f"  Агентов в workflow: {len(chat_info['workflow_chain']['agents'])}")


def example_6_health_check():
    """Проверка состояния API"""
    print("\n=== Пример 6: Health Check ===")
    
    response = requests.get(f"{BASE_URL}/health")
    health = response.json()
    
    print(f"API Status: {'[Online]' if health['success'] else '[Offline]'}")
    print(f"Message: {health['message']}")
    if health.get('data'):
        print(f"Version: {health['data'].get('version')}")


if __name__ == "__main__":
    print("="*60)
    print(" Graph Architect API - Примеры использования")
    print("="*60)
    print("\nУбедитесь что сервер запущен: python main.py")
    print("API доступен по адресу: http://localhost:8000")
    print("Документация: http://localhost:8000/docs")
    
    try:
        # Проверяем доступность API
        example_6_health_check()
        
        # Запускаем примеры
        example_1_simple_message()
        example_2_streaming()
        example_3_workflow()
        example_4_document_upload()
        example_5_chat_management()
        
        print("\n" + "="*60)
        print("[OK] All examples completed successfully!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Ошибка: Не удается подключиться к API")
        print("Запустите сервер: python main.py")
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
