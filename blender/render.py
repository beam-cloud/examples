import bpy
import sys
import json

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

output_path = argv[0]
json_path = argv[1]

with open(json_path, "r") as f:
    config = json.load(f)


def create_material(name, color):
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
    else:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    mat.node_tree.nodes.clear()

    bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)

    output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


if "material_color" in config:
    color = config["material_color"]
    new_material = create_material("TreeMaterial", color)

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj_name_lower = obj.name.lower()

            if any(
                keyword in obj_name_lower
                for keyword in ["tree", "leaf", "leaves", "foliage"]
            ):
                obj.data.materials.clear()
                obj.data.materials.append(new_material)

if "object_to_modify" in config and "location" in config:
    obj = bpy.data.objects.get(config["object_to_modify"])
    if obj:
        obj.location = config["location"]

scene = bpy.context.scene
scene.render.filepath = output_path
scene.render.image_settings.file_format = "PNG"

try:
    scene.render.engine = "BLENDER_EEVEE"
except:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32

bpy.ops.render.render(write_still=True)
