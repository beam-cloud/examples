from pathlib import Path
import os
import time
import slack_sdk
import importlib
from beam import schedule, Image


def _create_config(token: str):
    # This is required to be added so beta9 cli wont prompt for token
    config_path: Path = Path("~/.beta9/config.ini").expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w+") as config_file:
        config_file.write(f"[default]\ntoken = {token}\n")
        config_file.write("gateway_host = gateway.stage.beam.cloud\n")
        config_file.write("gateway_port = 443\n")


def _create_beta9_script():
    # Create a script that can be run from the command line
    script_path: Path = Path("/usr/local/bin/beam")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w+") as script_file:
        script_file.write("#!/bin/bash\n")
        script_file.write('python -m beta9 "$@"\n')
    script_path.chmod(777)


@schedule(
    "0 2,14,19 * * 1-5",
    app="examples",
    secrets=[
        "BEAM_WORKSPACE_ID",
        "BEAM_AUTH_TOKEN",
        "BEAM_GATEWAY_HOST",
        "BEAM_API_HOST",
        "SLACK_WEBHOOK_URL",
    ],
    image=Image()
    .add_python_packages(
        ["slack-sdk", "pytest", "requests", "backoff", "numpy", "paramiko"]
    )
    .add_commands(["mkdir ~/.beta9"]),
)
def run_tests():
    _create_config(os.getenv("BEAM_AUTH_TOKEN"))
    config_path: Path = Path("~/.beta9/config.ini").expanduser()
    print(f"Config path: {config_path}")
    print(f"Config exists: {config_path.exists()}")

    container_id = os.environ["CONTAINER_ID"]

    _create_beta9_script()

    start_time = time.time()
    module = importlib.import_module("tests.light_test")
    module_funcs = [
        getattr(module, func) for func in dir(module) if callable(getattr(module, func))
    ]

    tests = [func for func in module_funcs if func.__name__.startswith("test_")]

    tests_index_failed = []
    test_times = []
    for i, test in enumerate(tests):
        test_time_start = time.time()
        try:
            del os.environ[
                "CONTAINER_ID"
            ]  # This is so that beta9 cli commands is allowed to execute new functions in a worker
            test()
        except BaseException as e:
            os.environ["CONTAINER_ID"] = container_id
            tests_index_failed.append(i)
            print(f"Test {test.__name__} failed with error: {e}")
        finally:
            os.environ["CONTAINER_ID"] = container_id
            test_times.append(time.time() - test_time_start)

    total_time = time.time() - start_time
    ship_results_to_slack(tests, tests_index_failed, test_times, total_time)


def ship_results_to_slack(tests, test_index_failed, test_times, total_time):
    slack_url = os.getenv("SLACK_WEBHOOK_URL")
    print(f"Slack url: {slack_url}")
    total_time_str = seconds_to_readable(total_time)

    test_results_str = ""
    for i, test in enumerate(tests):
        name = test.__name__
        emoji = "✅"

        if i in test_index_failed:
            emoji = "❌"

        time_string = seconds_to_readable(test_times[i])

        test_results_str += f"{emoji} {name}: {time_string}\n"

    blocks = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Integration Tests Results 🥽🔬",
                    "emoji": True,
                },
            },
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": test_results_str}},
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "plain_text",
                    "text": f"Results {len(tests) - len(test_index_failed)}/{len(tests)} succeeded. Took {total_time_str}",
                    "emoji": True,
                },
            },
        ]
    }

    client = slack_sdk.webhook.WebhookClient(slack_url)
    client.send_dict(blocks)


def seconds_to_readable(seconds):
    if seconds < 1:
        return f"{round(seconds * 1000, 3)} ms"

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_string = ""
    time_string += f"{int(hours)}h " if hours > 0 else ""
    time_string += f"{int(minutes)}m " if minutes > 0 else ""
    time_string += f"{round(seconds,3)}s"

    return time_string


if __name__ == "__main__":
    run_tests()
