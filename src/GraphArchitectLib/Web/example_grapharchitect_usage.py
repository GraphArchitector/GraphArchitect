"""
Examples of using Web API with GraphArchitect integration.

Demonstrates:
- Real tool selection via softmax
- Strategy finding with different algorithms
- Training based on feedback
- Getting tool metrics
"""

import requests
import json
import time


BASE_URL = "http://localhost:8000/api"


def example_1_health_check():
    """Example 1: Check integration status."""
    print("\n" + "="*70)
    print("EXAMPLE 1: GraphArchitect Integration Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    health = response.json()
    
    print(f"\nAPI Status: {' [Online]' if health['success'] else '[Offline]'}")
    print(f"Version: {health['data']['version']}")
    
    features = health['data']['features']
    print(f"\nOperating mode:")
    print(f"  GraphArchitect: {'[Activated]' if features['real_algorithms'] else '[Simulation]'}")
    print(f"  Real algorithms: {'[Yes]' if features['real_algorithms'] else '[No]'}")
    print(f"  Softmax selection: {'[Yes]' if features['softmax_selection'] else '[No]'}")
    print(f"  Training: {'[Yes]' if features['training'] else '[No]'}")
    print(f"  NLI: {'[Yes]' if features['nli'] else '[No]'}")
    
    return features['real_algorithms']


def example_2_streaming_with_real_algorithms():
    """Example 2: Streaming with real algorithms."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Task Execution with Real Algorithms")
    print("="*70)
    
    chat_id = f"demo_{int(time.time())}"
    
    # Test different algorithms
    algorithms = ["dijkstra", "yen_5", "ant_5"]
    
    for algo in algorithms:
        print(f"\n[Algorithm: {algo}]")
        print("-" * 70)
        
        response = requests.post(
            f"{BASE_URL}/chat/{chat_id}/message/stream",
            data={
                "message": "Analyze this text and determine its category",
                "planning_algorithm": algo
            },
            stream=True
        )
        
        print("\nEvents received:")
        
        event_count = 0
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    event_count += 1
                    
                    event_type = event.get('type')
                    
                    if event_type == 'agent_selected':
                        agent_id = event.get('agent_id', 'unknown')
                        score = event.get('score', 0)
                        metadata = event.get('metadata', {})
                        temp = metadata.get('temperature', 0)
                        logit = metadata.get('logit', 0)
                        
                        print(f"\n  [Tool Selected]")
                        print(f"    Tool ID:     {agent_id}")
                        print(f"    Probability: {score:.3f} (from softmax!)")
                        print(f"    Temperature: {temp:.3f}")
                        print(f"    Logit:       {logit:.3f}")
                    
                    elif event_type == 'text':
                        content = event.get('content', '')
                        if content:
                            print(f"\n  [Result]")
                            print(f"    {content[:200]}")
                
                except json.JSONDecodeError:
                    pass
        
        print(f"\n  Total events: {event_count}")


def example_3_training_feedback():
    """Example 3: Submit training feedback."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Training with User Feedback")
    print("="*70)
    
    # Check if Training Service is active
    response = requests.get(f"{BASE_URL}/training/statistics")
    
    if response.status_code != 200:
        print("  [WARNING] Training Service not active")
        return
    
    # Submit feedback
    task_id = f"task_{int(time.time())}"
    quality_score = 0.92
    
    print(f"\nSubmitting feedback:")
    print(f"  Task ID: {task_id}")
    print(f"  Quality: {quality_score}")
    
    response = requests.post(
        f"{BASE_URL}/training/feedback",
        data={
            "task_id": task_id,
            "quality_score": quality_score,
            "comment": "Excellent classification result"
        }
    )
    
    result = response.json()
    
    if result['success']:
        print(f"\n  [OK] Feedback submitted")
        print(f"    Tools updated: {result['data'].get('tools_updated', 0)}")
    else:
        print(f"\n  [ERROR] Feedback failed")


def example_4_get_tool_metrics():
    """Example 4: Get tool metrics."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Getting Tool Metrics")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/training/tools")
    
    if response.status_code != 200:
        print("  [WARNING] Metrics not available")
        return
    
    result = response.json()
    
    if 'tools' in result:
        tools = result['tools']
        
        print(f"\nTool metrics (Top 5 by reputation):")
        
        # Sort by reputation
        sorted_tools = sorted(tools, key=lambda x: x.get('reputation', 0), reverse=True)[:5]
        
        for tool in sorted_tools:
            print(f"\n  {tool['tool_name']}:")
            print(f"    Reputation:     {tool.get('reputation', 0):.3f}")
            print(f"    Sample size:    {tool.get('training_sample_size', 0)}")
            print(f"    Variance:       {tool.get('variance_estimate', 0):.3f}")
    else:
        print(f"  [ERROR] Failed to get metrics")


def example_5_compare_algorithms():
    """Example 5: Compare different pathfinding algorithms."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Algorithm Comparison")
    print("="*70)
    
    chat_id = f"compare_{int(time.time())}"
    message = "Find information and create a report"
    
    algorithms = [
        ("dijkstra", "Single best path"),
        ("yen_3", "Top-3 paths"),
        ("yen_5", "Top-5 paths"),
        ("ant_5", "Ant Colony (Top-5)")
    ]
    
    for algo, description in algorithms:
        print(f"\n[{algo}] - {description}")
        print("-" * 70)
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/{chat_id}/message/stream",
            data={
                "message": message,
                "planning_algorithm": algo
            },
            stream=True
        )
        
        strategies_count = 0
        selected_tools = []
        
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line)
                    
                    if event.get('type') == 'strategies_found':
                        strategies_count = event.get('count', 0)
                    
                    elif event.get('type') == 'agent_selected':
                        selected_tools.append(event.get('agent_id'))
                
                except json.JSONDecodeError:
                    pass
        
        elapsed = time.time() - start_time
        
        print(f"  Strategies found: {strategies_count}")
        print(f"  Tools selected:   {len(selected_tools)}")
        print(f"  Time:             {elapsed:.2f}s")


def example_6_get_specific_tool_metrics():
    """Example 6: Get metrics for specific tool."""
    agent_id = "agent-classifier-gpt4"
    
    print(f"\n[Getting metrics for: {agent_id}]")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/training/tools/{agent_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(result)
        
        if result['success']:
            tool = result['data']
            
            print(f"\n[OK] Tool metrics:")
            print(f"  Name:              {tool['tool_name']}")
            print(f"  Reputation:        {tool.get('reputation', 0):.3f}")
            print(f"  Training samples:  {tool.get('training_sample_size', 0)}")
            print(f"  Variance:          {tool.get('variance_estimate', 0):.3f}")
            print(f"  Mean cost:         ${tool.get('mean_cost', 0):.4f}")
            print(f"  Mean time:         {tool.get('mean_time_answer', 0):.2f}s")
            
            quality_scores = tool.get('quality_scores', [])
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"  Avg quality:       {avg_quality:.3f} (from {len(quality_scores)} scores)")
    else:
        print(f"  [ERROR] Tool not found (status {response.status_code})")


def main():
    """Run all examples."""
    print("="*70)
    print("GRAPHARCHITECT WEB API - USAGE EXAMPLES")
    print("="*70)
    print("\nMake sure server is running:")
    print("  python main.py")
    print("\n" + "="*70)
    
    try:
        ## Example 1: Health check
        #grapharchitect_enabled = example_1_health_check()
        
        #if not grapharchitect_enabled:
        #    print("\n[WARNING] GraphArchitect is in simulation mode")
        #    print("Real algorithms are not active.")
        #    print("\nTo enable:")
        #    print("  1. Ensure grapharchitect library is in PYTHONPATH")
        #    print("  2. Restart server")
        #    return
        
        ## Example 2: Streaming with algorithms
        #example_2_streaming_with_real_algorithms()
        
        ## Example 3: Training feedback
        ##example_3_training_feedback()
        
        ## Example 4: Get all tool metrics
        #example_4_get_tool_metrics()
        
        ## Example 5: Compare algorithms
        #example_5_compare_algorithms()
        
        ## Example 6: Specific tool metrics
        example_6_get_specific_tool_metrics()
        
        # Summary
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETE")
        print("="*70)
        print("\nKey points:")
        print("  1. GraphArchitect uses real algorithms (not random)")
        print("  2. Softmax selection with adaptive temperature")
        print("  3. Training updates tool reputation")
        print("  4. Metrics available via API")
        print("\n" + "="*70)
    
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to server")
        print("Make sure server is running: python main.py")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
