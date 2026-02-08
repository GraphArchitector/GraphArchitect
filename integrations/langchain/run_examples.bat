@echo off
echo ======================================================================
echo GraphArchitect + LangChain Integration - Examples
echo ======================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Install Python or add it to PATH
    pause
    exit /b 1
)

set PYTHON=python

echo Select example to run:
echo.
echo   1. Example 1: Basic Integration (5 min)
echo   2. Example 2: LangChain Agent with GraphArchitect tools (10 min)
echo   3. Example 3: Hybrid Execution (10 min)
echo   4. Run all examples
echo   0. Exit
echo.

set /p choice="Enter choice (0-4): "

if "%choice%"=="1" (
    echo.
    echo Running: Example 1 - Basic Integration
    echo ======================================================================
    cd examples
    "%PYTHON%" example_01_basic_integration.py
    cd ..
    echo.
    pause
)

if "%choice%"=="2" (
    echo.
    echo Running: Example 2 - LangChain Agent
    echo ======================================================================
    echo.
    
    if not defined OPENAI_API_KEY (
        echo [WARNING] OPENAI_API_KEY not set
        echo Example will use mock LLM
        echo.
        echo To use real LLM:
        echo   set OPENAI_API_KEY=your-key
        echo.
        pause
    )
    
    cd examples
    "%PYTHON%" example_02_langchain_agent.py
    cd ..
    echo.
    pause
)

if "%choice%"=="3" (
    echo.
    echo Running: Example 3 - Hybrid Execution
    echo ======================================================================
    cd examples
    "%PYTHON%" example_03_hybrid_execution.py
    cd ..
    echo.
    pause
)

if "%choice%"=="4" (
    echo.
    echo Running all examples...
    echo.
    
    echo [1/3] Basic Integration
    echo ======================================================================
    cd examples
    "%PYTHON%" example_01_basic_integration.py
    cd ..
    echo.
    pause
    
    echo.
    echo [2/3] LangChain Agent
    echo ======================================================================
    cd examples
    "%PYTHON%" example_02_langchain_agent.py
    cd ..
    echo.
    pause
    
    echo.
    echo [3/3] Hybrid Execution
    echo ======================================================================
    cd examples
    "%PYTHON%" example_03_hybrid_execution.py
    cd ..
    echo.
    
    echo.
    echo ======================================================================
    echo All examples completed!
    echo ======================================================================
    pause
)

if "%choice%"=="0" (
    exit /b 0
)

if not "%choice%"=="1" if not "%choice%"=="2" if not "%choice%"=="3" if not "%choice%"=="4" if not "%choice%"=="0" (
    echo.
    echo Invalid choice. Run again and select 0-4.
    pause
)
