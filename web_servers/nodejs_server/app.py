from beam import Image, Pod

server = Pod(
    image=Image(base_image="node:latest", python_version="python3.12").add_commands(
        [
            "git clone https://github.com/beam-cloud/examples /tmp/examples",
            "cd /tmp/examples/web_servers/nodejs_server/app && npm install",
        ]
    ),
    ports=[3000],
    cpu=1,
    memory=1024,
    entrypoint=["node", "/tmp/examples/web_servers/nodejs_server/app/index.js"],
)

res = server.create()

print("✨ Server hosted at:", res.url)
