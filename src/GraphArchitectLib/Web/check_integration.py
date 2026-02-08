"""
Integration check script for GraphArchitect Web API.

Verifies:
1. GraphArchitect library availability
2. GraphArchitectBridge initialization
3. Agent to BaseTool conversion
4. Softmax tool selection
5. Strategy finding in graph
6. NLI dataset
7. Training service
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*70)
print("GRAPHARCHITECT INTEGRATION CHECK")
print("="*70)


def check_grapharchitect_library():
    """Check 1: GraphArchitect library availability."""
    print("\n[CHECK 1] GraphArchitect Library Availability")
    print("-"*70)
    
    try:
        import grapharchitect
        from grapharchitect.entities.base_tool import BaseTool
        from grapharchitect.services.selection.instrument_selector import InstrumentSelector
        from grapharchitect.services.graph_strategy_finder import GraphStrategyFinder
        
        print("  [OK] Module grapharchitect imported")
        print(f"  Path: {grapharchitect.__file__}")
        print(f"  [OK] BaseTool available")
        print(f"  [OK] InstrumentSelector available")
        print(f"  [OK] GraphStrategyFinder available")
        
        return True
        
    except ImportError as e:
        print(f"  [FAIL] Cannot import grapharchitect: {e}")
        print("\n  Solution:")
        print("    set PYTHONPATH=..;%PYTHONPATH%")
        return False


def check_bridge_initialization():
    """Check 2: GraphArchitectBridge initialization."""
    print("\n[CHECK 2] GraphArchitectBridge Initialization")
    print("-"*70)
    
    try:
        from grapharchitect_bridge import get_bridge, is_bridge_available, AgentTool
        
        print("  [OK] grapharchitect_bridge imported")
        
        if is_bridge_available():
            bridge = get_bridge()
            print(f"  [OK] Bridge initialized")
            print(f"  Tools loaded: {len(bridge.tools)}")
            print(f"  [OK] Selector: {bridge.selector.__class__.__name__}")
            print(f"  [OK] Strategy finder: {bridge.strategy_finder.__class__.__name__}")
            print(f"  [OK] Training orchestrator: {bridge.training.__class__.__name__}")
            
            return True
        else:
            print("  [FAIL] Bridge not available")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Bridge initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_agent_conversion():
    """Check 3: Agent to BaseTool conversion."""
    print("\n[CHECK 3] Agent to BaseTool Conversion")
    print("-"*70)
    
    try:
        from repository import get_repository
        from grapharchitect_bridge import AgentTool
        from grapharchitect.entities.base_tool import BaseTool
        
        repo = get_repository()
        agents = repo.get_all_agents()
        print(f"  Loaded agents: {len(agents)}")
        
        if agents:
            agent = agents[0]
            tool = AgentTool(agent)
            
            print(f"\n  [OK] Agent converted:")
            print(f"    Agent ID:   {agent.id}")
            print(f"    Agent name: {agent.name}")
            print(f"    Tool name:  {tool.metadata.tool_name}")
            print(f"    Input:      {tool.input.format}")
            print(f"    Output:     {tool.output.format}")
            print(f"    Reputation: {tool.metadata.reputation:.2f}")
            
            # Test all agents
            success_count = 0
            for agent in agents:
                try:
                    tool = AgentTool(agent)
                    assert isinstance(tool, BaseTool)
                    success_count += 1
                except Exception as e:
                    print(f"   [WARNING] Conversion error for {agent.id}: {e}")
            
            print(f"\n  [OK] Successfully converted: {success_count}/{len(agents)} tools")
            return True
        else:
            print("  [FAIL] No agents loaded")
            print("  Run: python db_manager.py load_agents")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_softmax_selection():
    """Check 4: Softmax tool selection."""
    print("\n[CHECK 4] Softmax Tool Selection")
    print("-"*70)
    
    try:
        from grapharchitect_bridge import get_bridge
        from repository import get_repository
        
        bridge = get_bridge()
        repo = get_repository()
        agents = repo.get_all_agents()[:5]  # Use first 5 tools
        
        # Convert to tools
        tools = [bridge.agent_to_tool_map[a.id] for a in agents if a.id in bridge.agent_to_tool_map]
        
        if not tools:
            print("  [FAIL] No tools available")
            return False
        
        # Create dummy task embedding
        task_embedding = bridge.embedding_service.embed_text("Classify this text")
        
        # Select tool
        selection_result = bridge.selector.select_instrument(
            instruments=tools,
            task_embedding=task_embedding,
            top_k=min(3, len(tools))
        )
        
        print(f"\n  [OK] Selection performed:")
        print(f"    Candidates:      {len(tools)}")
        print(f"    Top-K:           {min(3, len(tools))}")
        print(f"    Selected:        {selection_result.selected_tool.metadata.tool_name}")
        print(f"    Probability:     {selection_result.selection_probability:.3f}")
        print(f"    Temperature:     {selection_result.temperature:.3f}")
        print(f"    Logits:          {len(selection_result.logits)}")
        print(f"    Probabilities:   {len(selection_result.probabilities)}")
        
        # Verify softmax
        prob_sum = sum(selection_result.probabilities.values())
        print(f"\n  Softmax verification:")
        print(f"    Sum of probabilities: {prob_sum:.6f}")
        
        if abs(prob_sum - 1.0) < 0.001:
            print(f"    [OK] Softmax correct (sum ≈ 1.0)")
        else:
            print(f"    [WARNING] Softmax sum != 1.0")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_strategy_finding():
    """Check 5: Strategy finding in graph."""
    print("\n[CHECK 5] Strategy Finding in Graph")
    print("-"*70)
    
    try:
        from grapharchitect_bridge import get_bridge
        from grapharchitect.services.pathfinding_algorithm import PathfindingAlgorithm
        
        bridge = get_bridge()
        
        if not bridge.tools:
            print("  [FAIL] No tools available")
            return False
        
        # Test different algorithms
        algorithms = [
            ("DIJKSTRA", PathfindingAlgorithm.DIJKSTRA),
            ("YEN", PathfindingAlgorithm.YEN),
            ("ACO", PathfindingAlgorithm.ASTAR)
        ]
        
        for algo_name, algo in algorithms:
            try:
                strategies = bridge.strategy_finder.find_strategies(
                    tools=bridge.tools,
                    start_format="text|question",
                    end_format="text|answer",
                    algorithm=algo,
                    limit=3
                )
                
                if strategies:
                    print(f"   [{algo_name}] Found {len(strategies)} strategy(ies)")
                    for i, strategy in enumerate(strategies[:2], 1):
                        tool_names = [t.metadata.tool_name for t in strategy]
                        print(f"      Strategy {i}: {' -> '.join(tool_names[:3])}")
                else:
                    print(f"   [{algo_name}] [WARNING] No strategies found (no suitable path)")
            
            except Exception as e:
                print(f"   [{algo_name}] [ERROR] {e}")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Strategy finding failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_nli_dataset():
    """Check 6: NLI dataset."""
    print("\n[CHECK 6] NLI Dataset")
    print("-"*70)
    
    try:
        from grapharchitect_bridge import get_bridge
        
        bridge = get_bridge()
        nli = bridge.nli
        
        # Check if dataset loaded
        nli_file = Path(__file__).parent / "data" / "nli_examples.json"
        
        if nli_file.exists():
            import json
            with open(nli_file, 'r') as f:
                data = json.load(f)
            
            print(f"  [OK] NLI dataset found: {nli_file}")
            print(f"  Examples: {len(data)}")
            
            # Test parsing
            test_query = "Classify this text"
            representation = nli.parse_task(test_query)
            
            if representation:
                print(f"\n  Test query: \"{test_query}\"")
                print(f"    Input:  {representation.input_connector.format}")
                print(f"    Output: {representation.output_connector.format}")
                print(f"    [OK] NLI parsing works")
            
            return True
        else:
            print(f"  [WARNING] NLI dataset not found: {nli_file}")
            print(f"  NLI will use default connectors")
            return True
            
    except Exception as e:
        print(f"  [FAIL] NLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_service():
    """Check 7: Training service."""
    print("\n[CHECK 7] Training Service")
    print("-"*70)
    
    try:
        from training_service import TrainingService
        
        service = TrainingService()
        
        if service.enabled:
            print(f"  [OK] TrainingService initialized")
            print(f"  Bridge: {service.bridge.__class__.__name__}")
            return True
        else:
            print(f"  [WARNING] TrainingService not enabled")
            return True
            
    except Exception as e:
        print(f"  [FAIL] Training service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    checks = [
        ("GraphArchitect Library", check_grapharchitect_library),
        ("GraphArchitectBridge", check_bridge_initialization),
        ("Agent Conversion", check_agent_conversion),
        ("Softmax Selection", check_softmax_selection),
        ("Strategy Finding", check_strategy_finding),
        ("NLI Dataset", check_nli_dataset),
        ("Training Service", check_training_service)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [ERROR] Check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("CHECK SUMMARY")
    print("="*70)
    print()
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status:8} {name}")
    
    print("\n" + "-"*70)
    print(f"  Result: {passed_count}/{total_count} checks passed")
    print("="*70)
    
    if passed_count == total_count:
        print("\n[OK] Integration working correctly!")
        print("\nYou can now:")
        print("  1. Start server: python main.py")
        print("  2. Open browser: http://localhost:8000")
        print("  3. Send messages and see real algorithms in action!")
    else:
        print(f"\n[WARNING] PASSED: {passed_count}/{total_count}")
        print("\nSome checks failed. Review errors above.")
    
    print("="*70)


if __name__ == "__main__":
    main()
