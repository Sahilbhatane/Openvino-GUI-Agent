import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"


def send_task(base_url: str, instruction: str) -> dict:
    url = f"{base_url}/run-task"
    payload = json.dumps({"instruction": instruction}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=600)
    return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a GUI task to the OpenVINO GUI Agent server."
    )
    parser.add_argument(
        "instruction",
        help='Natural-language task, e.g. "open calculator"',
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of the agent server (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    print(f"Sending task: {args.instruction!r}")
    print(f"Server:       {args.url}")
    print("Waiting for agent response ...\n")

    try:
        result = send_task(args.url, args.instruction)
    except urllib.error.URLError:
        print("ERROR: Agent server is not running.")
        print("Start it with:  python main.py")
        sys.exit(1)

    print(f"Status:     {result.get('status')}")
    print(f"Iterations: {result.get('iterations')}")

    usage = result.get("token_usage", {})
    if usage:
        print(
            f"Tokens:     {usage.get('total_input_tokens', 0)} in / "
            f"{usage.get('total_output_tokens', 0)} out  "
            f"({usage.get('total_generation_time', 0)}s, "
            f"{usage.get('avg_tokens_per_second', 0)} tok/s)"
        )
    print()

    for step in result.get("history", []):
        it = step.get("iteration", "?")
        thought = step.get("thought", "")
        actions = step.get("actions", [])
        results = step.get("results", [])
        done = step.get("task_complete", False)

        tok_str = ""
        if step.get("input_tokens"):
            tok_str = (
                f"  [{step['input_tokens']} in / {step.get('output_tokens', 0)} out"
                f"  {step.get('generation_time', 0)}s"
                f"  {step.get('tokens_per_second', 0)} tok/s]"
            )

        print(f"--- Iteration {it} ---{tok_str}")
        print(f"  Thought: {thought}")
        if done:
            print("  Task complete.")
        for act, res in zip(actions, results):
            desc = act.get("description") or act.get("type", "")
            print(f"  Action: {desc}  ->  {res}")
        print()

    if result.get("status") == "completed":
        print("Task completed successfully.")
    else:
        print("Task did not complete within the iteration limit.")


if __name__ == "__main__":
    main()
