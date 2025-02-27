from beam import Image, Pod

server = Pod(
    image=Image(base_image="node:latest", python_version="python3.12").add_commands(
        [
            "git clone https://github.com/NilayYadav/pod-servers /tmp/pod-servers",
            "cd /tmp/pod-servers/nodejs && npm install",
        ]
    ),
    ports=[3000],
    cpu=1,
    memory=1024,
    entrypoint=["node", "/tmp/pod-servers//nodejs/index.js"],
)

res = server.create()

print("✨ Server hosted at:", res.url)
