"""
A simple VSCode server that runs in a remote environment using Pod. Access the full
VSCode experience directly from your browser, anywhere.

To deploy: python app.py
"""

from beam import Pod, Image

OPENVSCODE_SERVER_VERSION = "1.97.2"
vscode_port = 8080

image = Image("python3.12").add_commands(
    [
        f"wget https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-v{OPENVSCODE_SERVER_VERSION}/openvscode-server-v{OPENVSCODE_SERVER_VERSION}-linux-x64.tar.gz",
        f"tar -xzf openvscode-server-v{OPENVSCODE_SERVER_VERSION}-linux-x64.tar.gz",
        f"mv openvscode-server-v{OPENVSCODE_SERVER_VERSION}-linux-x64 /opt/openvscode-server",
        f"rm openvscode-server-v{OPENVSCODE_SERVER_VERSION}-linux-x64.tar.gz",
        "mkdir -p /root/.local/share/openvscode-server/extensions",
        "chmod +x /opt/openvscode-server/bin/openvscode-server",
    ]
)

vscode_server = Pod(
    image=image,
    ports=vscode_port,
    cpu=4,
    memory="8Gi",
    entrypoint=[
        "/opt/openvscode-server/bin/openvscode-server",
        "--host",
        "::",
        "--port",
        str(vscode_port),
        "--without-connection-token",
        "--disable-workspace-trust",
        "--telemetry-level",
        "off",
        "/root/app",
    ],
)

res = vscode_server.create()
print("Access your VSCode server at: ", res.url)
