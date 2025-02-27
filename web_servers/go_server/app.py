from beam import Image, Pod

server = Pod(
    image=Image(base_image="golang:latest", python_version="python3.12").add_commands(
        [
            "git clone https://github.com/NilayYadav/pod-servers /tmp/pod-servers",
        ]
    ),
    ports=[8080],
    cpu=1,
    memory=1024,
    entrypoint=["go", "run", "/tmp/pod-servers/go/main.go"],
)

res = server.create()

print("✨ Server hosted at:", res.url)
