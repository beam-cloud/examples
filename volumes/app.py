from beam import function, Volume


VOLUME_PATH = "./example-volume"


@function(
    volumes=[Volume(name="example-volume", mount_path=VOLUME_PATH)],
)
def access_files():
    # Write files to a volume
    with open(f"{VOLUME_PATH}/somefile.txt", "w") as f:
        f.write("On the volume!")

    # Read files from a volume
    s = ""
    with open(f"{VOLUME_PATH}/somefile.txt", "r") as f:
        s = f.read()

    return s


if __name__ == "__main__":
    print(access_files.remote())
