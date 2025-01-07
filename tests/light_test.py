import subprocess
import pytest
import os
import re
import json
import requests
import backoff

WORKSPACE_ID = os.getenv("BEAM_WORKSPACE_ID")
AUTH_TOKEN = os.getenv("BEAM_AUTH_TOKEN")
GATEWAY_HOST = os.getenv("BEAM_GATEWAY_HOST", "app.beam.cloud")
API_HOST = os.getenv("BEAM_API_HOST", "api.beam.cloud")

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
        f"https://{GATEWAY_HOST}/api/v1/deployment/{WORKSPACE_ID}?name={deployment_name}&limit={100}",
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
                f"https://{GATEWAY_HOST}/api/v1/deployment/{WORKSPACE_ID}/{d_id}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + AUTH_TOKEN,
                },
            )

def prepare_app_path(directory):
    file_name = "app.py"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    app_path = os.path.join(parent_dir, directory, file_name)
    os.chdir(os.path.dirname(app_path))
    test_name = os.path.join(directory, file_name).replace("\\", "/")
    return test_name, app_path, current_dir

def test_quickstart():
    dir_name = "quickstart"
    test_name, _, current_dir = prepare_app_path(dir_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:predict",
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
            assert "result" in response.text, f"{test_name} unexpected response"
    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_custom_image():
    dir_name = "custom_images"
    test_name, _, current_dir = prepare_app_path(dir_name)
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:handler",
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
    test_name, _, current_dir = prepare_app_path("gpu_acceleration")
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:handler",
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
    test_name, app_path, current_dir = prepare_app_path("secrets")

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
    test_name, _, current_dir = prepare_app_path("endpoints")
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:multiply",
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
    test_name, _, current_dir = prepare_app_path("keep_warm")
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:handler",
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
    test_name, _, current_dir = prepare_app_path("preload_models")
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:predict",
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
        
        retries = 0
        while True:
            try:
                with requests.Session() as session:
                    r.prepare_body(data=json.dumps({"prompt": "Hello, world!"}), files=None)
                    response = session.send(r, timeout=None)
                    assert (
                        response.status_code == 200
                    ), f"{test_name} request to endpoint failed with status code: {response.status_code} and response: {response.text}"
                    assert (
                        "prediction" in response.text
                    ), f"{test_name} prediction not found in output"
                    return
            except BaseException as e:
                retries += 1
                if retries > 3:
                    raise e

    finally:
        delete_deployments(deployment_name)
        os.chdir(current_dir)


def test_task_queue():
    test_name, _, current_dir = prepare_app_path("task_queues")
    deployment_name = test_name.split("/")[-1]

    try:
        command = [
            "beam",
            "deploy",
            "app.py:multiply",
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


            @backoff.on_predicate(backoff.expo, predicate=lambda x: x['status'] in ["PENDING", "RUNNING", "COMPLETED"] , max_tries=10)
            def get_task_status(task_id, auth_token):
                response = requests.get(
                    f"https://{API_HOST}/v2/task/{task_id}/",
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
    test_name, app_path, current_dir = prepare_app_path("callbacks")

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
    test_name, app_path, current_dir = prepare_app_path("functions")

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
    test_name, app_path, current_dir = prepare_app_path("scaling_out")

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
    test_name, app_path, current_dir = prepare_app_path("sharing_state")

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
    test_name, app_path, current_dir = prepare_app_path("volumes")

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
    test_name, app_path, current_dir = prepare_app_path("outputs")

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