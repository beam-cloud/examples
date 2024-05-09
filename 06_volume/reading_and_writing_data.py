"""
** Reading and Writing Data to Volumes **

You can read and write to your Beam Volumes like any ordinary Python file.
"""

from beam import function, Volume


VOLUME_PATH = "./model_weights"


@function(
    volumes=[Volume(name="model-weights", mount_path=VOLUME_PATH)],
)
def access_files():
    # Write files to a volume
    with open(f"{VOLUME_PATH}/somefile.txt", "w") as f:
        f.write("Writing to the volume!")

    # Read files from a volume
    with open(f"{VOLUME_PATH}/somefile.txt", "r") as f:
        f.read()
