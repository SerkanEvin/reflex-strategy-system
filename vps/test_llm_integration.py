#!/usr/bin/env python3
"""
Test script for Synthetic LLM API integration.
Tests the LLM client with various game state scenarios.
"""
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vps.llm_client import LLMClient
from shared.models import SecurityTier, DetectionType


# Create game state scenarios for testing
def create_test_state(scenario: str) -> dict:
    """Create test game state for the given scenario."""
    base_state = {
        "timestamp": datetime.now().timestamp(),
        "player": {
            "health_percent": 100.0,
            "is_in_combat": False,
            "coordinates": {"x": 0, "y": 0, "z": 0},
            "position": {"x": 0, "y": 0, "z": 0},
            "inventory_full_percent": 50.0
        },
        "security_tier": SecurityTier.SECURE.name,
        "detections": [],
        "active_targets": [],
        "frame_id": 0
    }

    scenarios = {
        "safe_no_enemies": {
            **base_state,
            "player": {
                **base_state["player"],
                "health_percent": 100.0,
                "inventory_full_percent": 20.0
            },
            "detections": [],
            "security_tier": SecurityTier.SECURE.name
        },
        "critical_health": {
            **base_state,
            "player": {
                **base_state["player"],
                "health_percent": 15.0,
                "is_in_combat": True
            },
            "detections": [
                {
                    "bbox": {"center": [10, 20, 0], "size": [50, 100]},
                    "detection_type": DetectionType.ENEMY.value,
                    "label": "Enemy",
                    "confidence": 0.95,
                    "distance": 15.0,
                    "health_percent": 80.0
                }
            ],
            "security_tier": SecurityTier.DANGER.name
        },
        "inventory_full": {
            **base_state,
            "player": {
                **base_state["player"],
                "health_percent": 80.0,
                "inventory_full_percent": 95.0
            },
            "detections": [],
            "security_tier": SecurityTier.SECURE.name
        },
        "resources_detected": {
            **base_state,
            "player": {
                **base_state["player"],
                "health_percent": 90.0,
                "inventory_full_percent": 30.0
            },
            "detections": [
                {
                    "bbox": {"center": [25, 40, 0], "size": [30, 30]},
                    "detection_type": DetectionType.RESOURCE.value,
                    "label": "Gold Ore",
                    "confidence": 0.88,
                    "distance": 45.0
                },
                {
                    "bbox": {"center": [80, 120, 0], "size": [25, 25]},
                    "detection_type": DetectionType.RESOURCE.value,
                    "label": "Iron Ore",
                    "confidence": 0.92,
                    "distance": 85.0
                }
            ],
            "security_tier": SecurityTier.SECURE.name
        },
        "enemies_near_safe_zone": {
            **base_state,
            "player": {
                **base_state["player"],
                "health_percent": 75.0,
                "inventory_full_percent": 50.0
            },
            "detections": [
                {
                    "bbox": {"center": [30, 50, 0], "size": [60, 120]},
                    "detection_type": DetectionType.ENEMY.value,
                    "label": "Goblin",
                    "confidence": 0.90,
                    "distance": 65.0,
                    "health_percent": 60.0
                }
            ],
            "security_tier": SecurityTier.SECURE.name
        }
    }

    return scenarios.get(scenario, base_state)


def create_test_context() -> dict:
    """Create test strategy context."""
    return {
        "spatial_memory": [
            {
                "type": "market",
                "label": "Grand Bazaar",
                "position": {"x": 500, "y": 200, "z": 0},
                "distance": 538.0
            },
            {
                "type": "spawn_point",
                "label": "Safe Haven",
                "position": {"x": 0, "y": 0, "z": 0},
                "distance": 0
            },
            {
                "type": "vendor",
                "label": "Equipment Shop",
                "position": {"x": 100, "y": 50, "z": 0},
                "distance": 111.0
            }
        ],
        "inventory_state": {
            "full_percent": 50.0,
            "items": []
        },
        "current_strategy": None
    }


async def test_scenario(
    client: LLMClient,
    scenario_name: str,
    scenario_state: dict,
    scenario_context: dict
) -> dict:
    """Test a single scenario."""
    print(f"\n{'='*60}")
    print(f"Testing Scenario: {scenario_name}")
    print(f"{'='*60}")

    # Build prompt
    prompt = client.get_strategy_prompt(scenario_state, scenario_context)

    # Add scenario-specific context
    print(f"Player Health: {scenario_state['player']['health_percent']}%")
    print(f"Inventory Full: {scenario_state['player']['inventory_full_percent']}%")
    print(f"Security Tier: {scenario_state['security_tier']}")
    print(f"Detections: {len(scenario_state['detections'])}")
    for det in scenario_state['detections']:
        print(f"  - {det['label']} ({det['detection_type']}) at {det['distance']:.1f}m")

    print()

    # Measure latency
    start_time = asyncio.get_event_loop().time()
    try:
        response = await client.chat_completion(
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )
        latency = (asyncio.get_event_loop().time() - start_time) * 1000

        # Print results
        print(f"Response received in {latency:.0f}ms")
        print(f"Action: {response.action}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Priority: {response.priority}")
        if response.target:
            print(f"Target: {response.target}")
        print(f"Tokens Used: {response.tokens_used}")
        print(f"Model: {response.model}")

        return {
            "scenario": scenario_name,
            "success": True,
            "action": response.action,
            "reasoning": response.reasoning,
            "priority": response.priority,
            "target": response.target,
            "latency_ms": round(latency, 2),
            "tokens_used": response.tokens_used
        }

    except Exception as e:
        latency = (asyncio.get_event_loop().time() - start_time) * 1000
        print(f"FAILED: {e}")
        return {
            "scenario": scenario_name,
            "success": False,
            "error": str(e),
            "latency_ms": round(latency, 2)
        }


async def main():
    """Main test function."""
    API_KEY = "syn_4b011e593839090a79affd5348e4de31"
    API_URL = "https://api.synthetic.new/openai/v1"
    MODEL = "hf:deepseek-ai/DeepSeek-V3"

    print(f"{'='*60}")
    print("Synthetic LLM API Integration Test")
    print(f"{'='*60}")
    print(f"API URL: {API_URL}")
    print(f"Model: {MODEL}")
    print()

    # Create LLM client
    client = LLMClient(
        api_key=API_KEY,
        api_url=API_URL,
        model=MODEL,
        timeout=10.0
    )

    # Test connection first
    print("Testing API connection...")
    conn_test = await client.test_connection()
    if conn_test["success"]:
        print(f"Connection successful!")
        print(f"Latency: {conn_test['latency_ms']}ms")
        print(f"Available models: {conn_test['available_models']}")
    else:
        print(f"Connection failed: {conn_test['error']}")
        return

    # Test scenarios
    scenarios = [
        ("safe_no_enemies", "Safe area, no enemies - expect PATROL or EXPLORE"),
        ("critical_health", "Critical health in combat - expect RETREAT (but LLM may suggest HUNT given prompt context)"),
        ("inventory_full", "Inventory full - expect GO_TO_MARKET"),
        ("resources_detected", "Resources detected - expect GATHER"),
        ("enemies_near_safe_zone", "Enemies near safe zone - expect HUNT or PATROL")
    ]

    results = []
    context = create_test_context()

    for scenario_name, description in scenarios:
        state = create_test_state(scenario_name)
        result = await test_scenario(client, description, state, context)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / len(results)
    total_tokens = sum(r.get("tokens_used", {}).get("total_tokens", 0) for r in results)

    print(f"Total scenarios: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"Total tokens used: {total_tokens}")

    print("\nScenario Results:")
    for result in results:
        status = "✓" if result["success"] else "✗"
        action = result.get("action", "N/A")
        print(f"{status} {result['scenario'][:40]:40} -> {action}")

    # Expected actions validation
    expected_actions = {
        "Safe area, no enemies": ["PATROL", "EXPLORE"],
        "Critical health in combat": ["RETREAT"],
        "Inventory full": ["GO_TO_MARKET"],
        "Resources detected": ["GATHER"],
        "Enemies near safe zone": ["HUNT", "PATROL"]
    }

    print("\nValidation:")
    for result in results:
        if not result["success"]:
            continue
        scenario_desc = result["scenario"]
        action = result.get("action", "")
        expected = expected_actions.get(scenario_desc, [])
        if expected and action in expected:
            print(f"✓ {scenario_desc}: {action} matches expected {expected}")
        elif expected:
            print(f"⚠ {scenario_desc}: {action} (expected {expected})")

    # Clean up
    await client.close()

    # Save results
    with open("/tmp/llm_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/llm_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
