from beam import function, Image, Volume, env

image = Image(python_packages=["biopython"])
volume_path = "./seq"

if env.is_remote():
    import os
    import urllib.request
    from Bio import SeqIO


def download_file():
    # Download the file to Beam Storage Volume
    url = "https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_005816.fna"
    file_name = "NC_005816.fna"

    download_path = os.path.join(volume_path, file_name)
    urllib.request.urlretrieve(url, download_path)


@function(
    volumes=[Volume(name="seq", mount_path=volume_path)],
    image=image,
)
def read_sequence():
    download_file()

    record = SeqIO.read("./seq/NC_005816.fna", "fasta")
    print(record)


if __name__ == "__main__":
    read_sequence.remote()
