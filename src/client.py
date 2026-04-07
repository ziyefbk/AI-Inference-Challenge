"""
Platform interaction client.
Implements the contestant-side loop: register -> query -> ask -> inference -> submit.
"""

import os
import sys
import asyncio
import httpx
import argparse
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import run_inference


PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://10.0.0.1:8003")
TOKEN = os.environ.get("CONTESTANT_TOKEN", "your_secret_token")
TEAM_NAME = os.environ.get("CONTESTANT_NAME", "team_alpha")
CONTESTANT_PORT = int(os.environ.get("CONTESTANT_PORT", "9000"))


async def register(client: httpx.AsyncClient) -> bool:
    """Register with the platform."""
    try:
        resp = await client.post(
            f"{PLATFORM_URL}/register",
            json={"name": TEAM_NAME, "token": TOKEN},
        )
        if resp.status_code == 200:
            print(f"[Client] Registered successfully as {TEAM_NAME}")
            return True
        else:
            print(f"[Client] Registration failed: {resp.text}")
            return False
    except Exception as e:
        print(f"[Client] Registration error: {e}")
        return False


async def query_task(client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Query for available tasks."""
    try:
        resp = await client.post(
            f"{PLATFORM_URL}/query",
            json={"token": TOKEN},
        )

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            print(f"[Client] Query failed: {resp.text}")
            return None
    except Exception as e:
        print(f"[Client] Query error: {e}")
        return None


async def accept_task(client: httpx.AsyncClient, task_id: int, target_sla: str) -> Optional[Dict[str, Any]]:
    """Accept a task."""
    try:
        resp = await client.post(
            f"{PLATFORM_URL}/ask",
            json={
                "token": TOKEN,
                "task_id": task_id,
                "sla": target_sla,
            },
        )

        if resp.status_code == 200:
            result = resp.json()
            if result.get("status") == "accepted":
                return result.get("task")
            elif result.get("status") == "rejected":
                print(f"[Client] Task rejected: {result.get('reason')}")
            elif result.get("status") == "closed":
                print(f"[Client] Task closed")
        return None
    except Exception as e:
        print(f"[Client] Accept error: {e}")
        return None


async def submit_results(client: httpx.AsyncClient, task_data: Dict[str, Any]) -> bool:
    """Submit inference results."""
    try:
        resp = await client.post(
            f"{PLATFORM_URL}/submit",
            json={
                "user": {"name": TEAM_NAME, "token": TOKEN},
                "msg": task_data,
            },
        )

        if resp.status_code == 200:
            print(f"[Client] Submitted task successfully")
            return True
        else:
            print(f"[Client] Submit failed: {resp.text}")
            return False
    except Exception as e:
        print(f"[Client] Submit error: {e}")
        return False


async def process_task(client: httpx.AsyncClient, task: Dict[str, Any]) -> bool:
    """
    Process a single task: run inference and submit results.
    """
    messages = task.get("messages", [])
    if not messages:
        print("[Client] No messages in task")
        return False

    # Run inference
    print(f"[Client] Running inference on {len(messages)} messages...")
    results = run_inference(messages)

    # Build submission data
    task_data = {
        "overview": task.get("overview", {}),
        "messages": results,
    }

    # Submit
    return await submit_results(client, task_data)


async def main_loop():
    """
    Main client loop.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        # Register
        if not await register(client):
            print("[Client] Failed to register, exiting")
            return

        # Main loop
        consecutive_failures = 0
        while True:
            try:
                # Query for tasks
                task_overview = await query_task(client)

                if task_overview is None:
                    # No tasks available, wait and retry
                    await asyncio.sleep(0.5)
                    consecutive_failures += 1
                    if consecutive_failures % 20 == 0:
                        print(f"[Client] Waiting for tasks... ({consecutive_failures} queries)")
                    continue

                consecutive_failures = 0
                task_id = task_overview.get("task_id")
                target_sla = task_overview.get("target_sla")
                target_reward = task_overview.get("target_reward", 1.0)

                print(f"[Client] Found task {task_id} (SLA: {target_sla}, Reward: {target_reward})")

                # Accept task
                task = await accept_task(client, task_id, target_sla)

                if task is None:
                    await asyncio.sleep(0.1)
                    continue

                # Process and submit
                success = await process_task(client, task)

                if success:
                    print(f"[Client] Task {task_id} completed successfully")
                else:
                    print(f"[Client] Task {task_id} failed")

                # Small delay to avoid hammering
                await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                print("\n[Client] Interrupted, exiting...")
                break
            except Exception as e:
                print(f"[Client] Loop error: {e}")
                await asyncio.sleep(1)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Contestant Client")
    parser.add_argument("--token", default=None, help="Contestant token")
    parser.add_argument("--name", default=None, help="Team name")
    parser.add_argument("--platform-url", default=None, help="Platform URL")
    args = parser.parse_args()

    # Override globals if provided via args
    global TOKEN, TEAM_NAME, PLATFORM_URL
    if args.token:
        TOKEN = args.token
    if args.name:
        TEAM_NAME = args.name
    if args.platform_url:
        PLATFORM_URL = args.platform_url

    print(f"[Client] Starting with token={TOKEN[:8]}..., name={TEAM_NAME}")
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
