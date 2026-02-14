"""
Диагностический скрипт для выявления проблем с веб-API GraphArchitect.
"""
import sys
import os

print("=" * 60)
print("GRAPHARCHITECT DIAGNOSTICS")
print("=" * 60)

# 1. Check Python version
print(f"\n[1] Python version: {sys.version}")
if sys.version_info < (3, 8):
    print("   [WARNING] Python 3.8+ required")
else:
    print("   [OK] Version compatible")

# 2. Check dependencies
print("\n[2] Checking dependencies:")
dependencies = [
    "fastapi",
    "uvicorn", 
    "python-socketio",
    "pydantic",
    "aiofiles"
]

missing = []
for dep in dependencies:
    try:
        __import__(dep.replace("-", "_"))
        print(f"   [OK] {dep}")
    except ImportError:
        print(f"   [MISSING] {dep}")
        missing.append(dep)

if missing:
    print(f"\n   Install missing: pip install {' '.join(missing)}")

# 3. Check GraphArchitect library
print("\n[3] Checking GraphArchitect library:")
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import grapharchitect
    print(f"   [OK] grapharchitect available")
    print(f"   Path: {grapharchitect.__file__}")
except ImportError as e:
    print(f"   [ERROR] grapharchitect NOT AVAILABLE: {e}")
    print(f"   Add to PYTHONPATH: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

# 4. Check integration
print("\n[4] Checking integration:")
try:
    from grapharchitect_bridge import get_bridge, is_bridge_available
    print(f"   [OK] grapharchitect_bridge available")
    
    if is_bridge_available():
        print(f"   [OK] Bridge initialized")
        bridge = get_bridge()
        tools_count = len(bridge.tools)
        print(f"   Tools loaded: {tools_count}")
    else:
        print(f"   [WARNING] Bridge not available (simulation mode will be used)")
        
except Exception as e:
    print(f"   [ERROR] Bridge loading error: {e}")
    import traceback
    traceback.print_exc()

# 5. Check database
print("\n[5] Checking database:")
try:
    from database import Database
    db = Database()
    print(f"   [OK] database.py loaded")
    print(f"   DB file: {db.db_path}")
    
    if os.path.exists(db.db_path):
        size_kb = os.path.getsize(db.db_path) / 1024
        print(f"   [OK] Database exists ({size_kb:.1f} KB)")
    else:
        print(f"   [WARNING] Database not created")
        print(f"   Run: python db_manager.py init")
        
except Exception as e:
    print(f"   [ERROR] Database error: {e}")

# 6. Check ports
print("\n[6] Checking ports:")
import socket

def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

for port in range(8000, 8011):
    if check_port(port):
        print(f"   [BUSY] Port {port}")
    else:
        print(f"   [FREE] Port {port}")
        break

# 7. Check API keys
print("\n[7] Checking API keys:")
api_keys = {
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

for key_name, key_value in api_keys.items():
    if key_value:
        print(f"   [OK] {key_name}: set ({key_value[:10]}...)")
    else:
        print(f"   [NOT SET] {key_name} (fallback will be used)")

# 8. Test tool execution
print("\n[8] Testing tool execution:")
try:
    from repository import get_repository
    from grapharchitect_bridge import AgentTool
    
    repo = get_repository()
    agents = repo.get_all_agents()
    if agents:
        test_agent = agents[0]
        tool = AgentTool(test_agent)
        
        print(f"   Testing: {tool.metadata.tool_name}")
        result = tool.execute("Test query")
        print(f"   [OK] Result: {result[:100]}...")
    else:
        print(f"   [WARNING] No tools loaded")
        print(f"   Run: python db_manager.py load_agents")
        
except Exception as e:
    print(f"   [ERROR] Execution test failed: {e}")
    import traceback
    traceback.print_exc()

# SUMMARY
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

if missing:
    print("[ISSUES] Missing dependencies")
    print(f"   Install: pip install {' '.join(missing)}")
elif not os.path.exists("grapharchitect.db"):
    print("[WARNING] Database not created")
    print("   Run: python db_manager.py init")
    print("   Run: python db_manager.py load_agents")
else:
    print("[OK] ALL READY TO START!")
    print("\nStart command:")
    print("   python main.py")
    print("\nOpen in browser:")
    print("   http://localhost:8000")

print("=" * 60)
