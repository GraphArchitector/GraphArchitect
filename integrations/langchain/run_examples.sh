#chmod +x run_examples.sh
#./run_examples.sh

echo "======================================================================"
echo "GraphArchitect + LangChain Integration - Examples"
echo "======================================================================"
echo

# Проверка Python в PATH
if ! command -v python >/dev/null 2>&1; then
    echo "[ERROR] Python not found in PATH"
    echo "Install Python or add it to PATH"
    exit 1
fi

PYTHON=python

echo "Select example to run:"
echo
echo "  1. Example 1: Basic Integration (5 min)"
echo "  2. Example 2: LangChain Agent with GraphArchitect tools (10 min)"
echo "  3. Example 3: Hybrid Execution (10 min)"
echo "  4. Run all examples"
echo "  0. Exit"
echo

read -p "Enter choice (0-4): " choice

run_example_1() {
    echo
    echo "Running: Example 1 - Basic Integration"
    echo "======================================================================"
    (cd examples && $PYTHON example_01_basic_integration.py)
    echo
    read -p "Press Enter to continue..."
}

run_example_2() {
    echo
    echo "Running: Example 2 - LangChain Agent"
    echo "======================================================================"
    echo

    if [ -z "$OPENAI_API_KEY" ]; then
        echo "[WARNING] OPENAI_API_KEY not set"
        echo "Example will use mock LLM"
        echo
        echo "To use real LLM:"
        echo "  export OPENAI_API_KEY=your-key"
        echo
        read -p "Press Enter to continue..."
    fi

    (cd examples && $PYTHON example_02_langchain_agent.py)
    echo
    read -p "Press Enter to continue..."
}

run_example_3() {
    echo
    echo "Running: Example 3 - Hybrid Execution"
    echo "======================================================================"
    (cd examples && $PYTHON example_03_hybrid_execution.py)
    echo
    read -p "Press Enter to continue..."
}

case "$choice" in
    1)
        run_example_1
        ;;
    2)
        run_example_2
        ;;
    3)
        run_example_3
        ;;
    4)
        echo
        echo "Running all examples..."
        echo

        echo "[1/3] Basic Integration"
        run_example_1

        echo "[2/3] LangChain Agent"
        run_example_2

        echo "[3/3] Hybrid Execution"
        run_example_3

        echo
        echo "======================================================================"
        echo "All examples completed!"
        echo "======================================================================"
        ;;
    0)
        exit 0
        ;;
    *)
        echo
        echo "Invalid choice. Run again and select 0-4."
        ;;
esac
