"""
This script uses Blender to render a 3D scene. It downloads a .blend file, executes a Python script to render the scene,

To run this on Beam, you can run 'python app.py' in the terminal. Change the render_test.json and render.py files to your own use case.
"""

from beam import function, Image, Output
from pathlib import Path
import subprocess

blender_image = (
    Image(python_version="python3.11")
    .add_python_packages(["bpy"])
    .add_commands(
        [
            "apt update && apt install -y xorg libxkbcommon0",
            "wget -q https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz -O /tmp/blender.tar.xz",
            "tar -xf /tmp/blender.tar.xz -C /opt",
            "mv /opt/blender-* /opt/blender",
            "ln -s /opt/blender/blender /usr/local/bin/blender",
        ]
    )
)


@function(image=blender_image, cpu=12, memory="32Gi", gpu="A10G")
def render(script_content, json_content, output_name):
    blend_path = "/tmp/render_test.blend"
    script_path = "/tmp/render_alternative.py"
    json_path = "/tmp/render_test.json"
    output_path = f"/tmp/{output_name}"

    subprocess.run(
        "wget https://vnyiodyihbjosm4n.public.blob.vercel-storage.com/Tree-y3VjlYkkByWNSOol56jE307zHqaNsp.blend -O /tmp/Tree.blend",
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    blend_bytes = Path("/tmp/Tree.blend").read_bytes()

    Path(blend_path).write_bytes(blend_bytes)
    Path(script_path).write_text(script_content)
    Path(json_path).write_text(json_content)

    cmd = ["blender", blend_path, "-b", "-P", script_path, "--", output_path, json_path]

    subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")

    output_file = Output(path=output_path)
    output_file.save()
    public_url = output_file.public_url(expires=400)
    return {"output_url": public_url}


if __name__ == "__main__":
    output_url = render(
        script_content=Path("render.py").read_text(),
        json_content=Path("render_test.json").read_text(),
        output_name="output.png",
    )
    print(f"Output URL: {output_url}")
