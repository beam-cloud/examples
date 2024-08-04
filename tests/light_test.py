import subprocess
import pytest
import os
import re
import json
import requests
import backoff

WORKSPACE_ID = os.getenv("BEAM_WORKSPACE_ID")
AUTH_TOKEN = os.getenv("BEAM_AUTH_TOKEN")

curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

def parse_curl(curl_command):
    # Parse the URL
    url_match = re.search(r"'(https?://[^']+)'", curl_command)
    if not url_match:
        raise ValueError("URL not found in curl command")
    url = url_match.group(1)

    headers = {}
    header_matches = re.finditer(r"-H '([^:]+): ([^']+)'", curl_command)
    for match in header_matches:
        headers[match.group(1)] = match.group(2)

    data_match = re.search(r"-d '(\{.*\})'", curl_command)
    data = json.loads(data_match.group(1)) if data_match else None

    method = "POST" if "-X POST" in curl_command else "GET"

    req = requests.Request(method, url, headers=headers, json=data)
    prepared_req = req.prepare()

    return prepared_req


def delete_deployments(deployment_name):
    res = requests.get(
        f"https://app.beam.cloud/api/v1/deployment/{WORKSPACE_ID}?name={deployment_name}&limit={100}",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + AUTH_TOKEN,
        },
    )

    res = json.loads(res.text)
    if res:
        for d in res:
            d_id = d["external_id"]
            requests.delete(
                f"https://app.beam.cloud/api/v1/deployment/{WORKSPACE_ID}/{d_id}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + AUTH_TOKEN,
                },
            )

def prepare_app_path(directory, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    app_path = os.path.join(parent_dir, directory, filename)
    os.chdir(os.path.dirname(app_path))
    test_name = os.path.join(directory, filename).replace("\\", "/")
    return test_name, app_path, current_dir

def test_quickstart():
    file_name = "quickstart.py"
    test_name, _, current_dir = prepare_app_path("01_getting_started", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:run",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"

        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            assert "success" in response.text, f"{test_name} unexpected response"
    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_custom_image():
    file_name = "custom_image.py"
    test_name, app_path, current_dir = prepare_app_path("02_customizing_environment", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:handler",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"
        curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

        # get the curl command and invoke it
        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            assert "torch_version" in response.text, f"{test_name} torch not found in output"

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_gpu_acceleration():
    file_name = "gpu_acceleration.py"
    test_name, app_path, current_dir = prepare_app_path("02_customizing_environment", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:handler",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"
        curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

        # get the curl command and invoke it
        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            assert (
                "This container has a GPU attached" in response.text
            ), f"{test_name} GPU not found in output"

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_using_secrets():
    file_name = "using_secrets.py"
    test_name, app_path, current_dir = prepare_app_path("02_customizing_environment", file_name)

    try:
        result = subprocess.run(["python", app_path], capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert "secret" in result.stdout, f"{test_name} no sum in output"

    finally:
        os.chdir(current_dir)


def test_creating_endpoint():
    file_name = "creating_endpoint.py"
    test_name, _, current_dir = prepare_app_path("03_endpoint", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:multiply",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"
        curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

        # get the curl command and invoke it
        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            assert "result" in response.text, f"{test_name} result not found in output"

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_keep_warm():
    file_name = "keep_warm.py"
    test_name, _, current_dir = prepare_app_path("03_endpoint", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:handler",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"
        curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

        # get the curl command and invoke it
        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            assert "warm" in response.text, f"{test_name} warm not found in output"

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_preload_models():
    file_name = "preload_models.py"
    test_name, _, current_dir = prepare_app_path("03_endpoint", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:predict",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"
        curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

        # get the curl command and invoke it
        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            r.prepare_body(json={"prompt": "Hello, world!"}, files=None, data=None)
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            assert (
                "prediction" in response.text
            ), f"{test_name} prediction not found in output"

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_async_task():
    file_name = "async_task.py"
    test_name, _, current_dir = prepare_app_path("04_task_queue", file_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            f"{file_name}:multiply",
            "--name",
            deployment_name,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert "Deployed" in result.stdout, f"{test_name} failed to deploy"
        curl_pattern = r"(curl -X POST.+\n(\s*-H .+\n)*\s*-d \'{.*?}\')"

        # get the curl command and invoke it
        match = re.search(curl_pattern, result.stdout, re.DOTALL)
        assert match is not None, f"{test_name} no curl command found"
        r = parse_curl(match.group(0))
        with requests.Session() as session:
            r.prepare_body(json={"x": 2}, files=None, data=None)
            response = session.send(r)
            assert (
                response.status_code == 200
            ), f"{test_name} request to endpoint failed with status code: {response.status_code}"
            res = json.loads(response.text)
            assert (
                res["task_id"] is not None
            ), f"{test_name} did not get task id in response"


            @backoff.on_predicate(backoff.expo, predicate=lambda x: x['status'] == 'PENDING', max_tries=10)
            def get_task_status(task_id, auth_token):
                response = requests.get(
                    f"https://api.beam.cloud/v2/task/{task_id}/",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {auth_token}",
                    },
                )
                response.raise_for_status()
                return json.loads(response.text)

            try:
                res = get_task_status(res['task_id'], AUTH_TOKEN)
                assert res["status"] == "COMPLETE", f"{test_name} task did not complete successfully"
            except requests.exceptions.RequestException as e:
                assert False, f"{test_name} request to get task status failed: {str(e)}"

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_task_callbacks():
    file_name = "task_callbacks.py"
    test_name, app_path, current_dir = prepare_app_path("04_task_queue", file_name)

    try:
        command = ["python", app_path]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert (
            "Sending data to callback" in result.stdout
        ), f"{test_name} failed to send data to callback"
        assert (
            "Callback request took" in result.stdout
        ), f"{test_name} callback was not successful"

    finally:
        os.chdir(current_dir)


def test_running_functions():
    file_name = "running_functions.py"
    test_name, app_path, current_dir = prepare_app_path("05_function", file_name)

    try:
        command = ["python", app_path]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert "{'sum': 285}" in result.stdout, f"{test_name} sum not found in output"
        assert "{'sum': 14}" in result.stdout, f"{test_name} sum not found in output"

    finally:
        os.chdir(current_dir)


def test_scaling_out():
    file_name = "scaling_out.py"
    test_name, app_path, current_dir = prepare_app_path("05_function", file_name)

    try:
        command = ["python", app_path]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert "result" in result.stdout, f"{test_name} result not found in output"

    finally:
        os.chdir(current_dir)


def test_sharing_state():
    file_name = "sharing_state.py"
    test_name, app_path, current_dir = prepare_app_path("05_function", file_name)

    try:
        command = ["python", app_path]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert "eli" in result.stdout, f"{test_name} local pop not found in output"
        assert "daniel" in result.stdout, f"{test_name} remote pop not found in output"

    finally:
        os.chdir(current_dir)


def test_volume_use():
    file_name = "volume_use.py"
    test_name, app_path, current_dir = prepare_app_path("06_volume", file_name)

    try:
        command = ["python", app_path]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert (
            "On the volume!" in result.stdout
        ), f"{test_name} On the volume not found in output"

    finally:
        os.chdir(current_dir)


def test_outputs():
    file_name = "outputs.py"
    test_name, app_path, current_dir = prepare_app_path("07_outputs", file_name)

    try:
        command = ["python", app_path]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"{test_name} failed with error: {result.stderr}"
        assert (
            "Function complete" in result.stdout
        ), f"{test_name} function did not complete"
        assert "Output ID" in result.stdout, f"{test_name} failed to find output ID"
        assert (
            "Output Exists: True" in result.stdout
        ), f"{test_name} failed to confirm output exists"

    finally:
        os.chdir(current_dir)


if __name__ == "__main__":
    pytest.main([__file__])
