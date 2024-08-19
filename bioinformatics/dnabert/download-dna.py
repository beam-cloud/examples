from beam import function, Image, Volume, env
import os

# This packages will only get imported in the remote cloud runtime
if env.is_remote():
    from Bio import Entrez, SeqIO

BEAM_VOLUME_PATH = "./seq"  # Distributed cloud storage volume mounted to this path


@function(
    volumes=[
        Volume(name="seq", mount_path=BEAM_VOLUME_PATH)
    ],  # Distributed cloud storage volume
    image=Image(python_packages=["biopython"]),
)
# Download DNA dataset from GenBank
def download(accession_number):
    Entrez.email = "your.email@example.com"

    with Entrez.efetch(
        db="nucleotide", id=accession_number, rettype="gb", retmode="text"
    ) as handle:
        record = SeqIO.read(handle, "genbank")

    file_path = os.path.join(BEAM_VOLUME_PATH, f"{accession_number}.gb")
    SeqIO.write(record, open(file_path, "w"), "genbank")

    print(
        f"Sequence ID: {record.id}\nLength: {len(record.seq)}\nDescription: {record.description}"
    )


if __name__ == "__main__":
    # Download plasmid sequence: https://www.ncbi.nlm.nih.gov/nuccore/AE017046
    download.remote("AE017046.1.gb")
