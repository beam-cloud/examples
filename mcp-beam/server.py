import os
from typing import Any
import subprocess
import time

from mcp.server.fastmcp import FastMCP
import tempfile


server = FastMCP("beam-mcp-server")


def run_beam_command(command: list[str], venv_path: str = None) -> dict[str, Any]:
    """Run a Beam CLI command and return the result"""
    try:
        command = (
            [
                "/Library/Frameworks/Python.framework/Versions/3.11/bin/uv",  # Replace with your uv path run 'which uv;
                "run",
                f"--directory={venv_path}", # This is the path to the virtual environment where the Beam CLI and other dependencies are installed
            ]
            if venv_path
            else []
        ) + command
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(command),
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
            "command": " ".join(command),
        }


@server.tool()
async def deploy_beam_pod(absolute_path_to_app: str) -> dict[str, Any]:
    """
    Deploy a Beam Pod on Beam using the provided parameters.

    Args:
        absolute_path_to_app: The absolute path to the Beam Pod to deploy.

    Returns:
        A dictionary containing deployment results.

    Raises:
        Exception: If deployment fails for any reason.
    """
    venv_path = os.path.dirname(absolute_path_to_app)
    app_name = os.path.basename(absolute_path_to_app)
    try:
        result = run_beam_command(["python", f"{app_name}"], venv_path)
        return result
    except Exception as e:
        print(f"Failed to deploy Beam app: {e}")
        raise


@server.tool()
async def deploy_beam_app(
    absolute_path_to_app: str, functionName: str
) -> dict[str, Any]:
    """
    Deploy a Beam application using the provided parameters.

    Args:
        absolute_path_to_app: The absolute path to the Beam application to deploy.
        functionName: The name of the function to deploy.

    Returns:
        A dictionary containing deployment results.

    Raises:
        Exception: If deployment fails for any reason.
    """
    venv_path = os.path.dirname(absolute_path_to_app)
    app_name = os.path.basename(absolute_path_to_app)

    try:
        result = run_beam_command(
            ["beam", "deploy", f"{app_name}:{functionName}"], venv_path
        )
        return result
    except Exception as e:
        print(f"Failed to deploy Beam app: {e}")
        raise


@server.tool()
async def build_beam_app(python_code: str) -> str:
    """
    Build a Beam application by saving the provided Python code to the specified path.

    Args:
        python_code (str): The Python code to save.

    Returns:
        absolute_path_to_app (str): The absolute path to the saved Beam application.

    Raises:
        Exception: If building fails for any reason.
    """
    try:
        # Create a temporary app file path to store the Beam application
        temp_dir = tempfile.gettempdir()
        app_name = f"beam_app_{int(time.time())}.py"
        absolute_path_to_app = os.path.join(temp_dir, app_name)

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(absolute_path_to_app), exist_ok=True)

        # Write the provided Python code to the file
        with open(absolute_path_to_app, "w", encoding="utf-8") as file:
            file.write(python_code)

        print(f"Beam application saved successfully at {absolute_path_to_app}")
        return absolute_path_to_app
    except Exception as e:
        print(f"Failed to build Beam app: {e}")
        raise


@server.resource("beam://docs")
async def get_beam_docs() -> str:
    """
    Provides the content of the Beam documentation file (docs.txt).
    Used primarily for providing context to the LLM via the client.
    """
    try:
        # Ensure docs.txt is in the same directory as this script or provide full path
        script_dir = os.path.dirname(__file__)
        docs_path = os.path.join(script_dir, "docs.txt")
        with open(docs_path, "r", encoding="utf-8") as f:
            print(f"Providing content from {docs_path}")
            return f.read()
    except FileNotFoundError:
        print(f"docs.txt not found at expected path: {docs_path}")
        return "Error: Beam documentation file (docs.txt) not found."
    except Exception as e:
        print(f"Error reading docs.txt: {e}")
        return f"Error reading Beam documentation file: {str(e)}"


if __name__ == "__main__":
    server.run()
