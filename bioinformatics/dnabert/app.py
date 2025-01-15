from beam import function, Image, Volume, env, Output

# This packages will only get imported in the remote cloud runtime
if env.is_remote():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from Bio import SeqIO


CHECKPOINT = "RaphaelMourad/Mistral-DNA-v1-422M-hg38"  # Embedding model
BEAM_VOLUME_PATH = "./cached_models"  # Model is cached here
# https://www.ncbi.nlm.nih.gov/nuccore/AE017046
DNA_FILE_PATH = "./seq/AE017046.1.gb"
CHUNK_SIZE = 3000  # Run in batches of 3000 base pairs each


def read_dna_sequence(file_path):
    records = list(SeqIO.parse(file_path, "genbank"))
    return "".join([str(record.seq) for record in records if record.seq])


@function(
    secrets=["HF_TOKEN"],  # Huggingface API key for embedding model
    name="dnabert",
    cpu=4,
    memory="32Gi",
    image=Image(
        python_version="python3.11",
        python_packages=[
            "transformers",
            "sentencepiece==0.1.99",
            "accelerate==0.23.0",
            "torch",
            "biopython",
            "einops",
            "huggingface_hub[hf-transfer]",
        ],
    ).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    volumes=[
        Volume(
            name="cached_models", mount_path=BEAM_VOLUME_PATH
        ),  # Embedding model is cached here
        # Path with the GenBank downloads
        Volume(name="seq", mount_path="./seq"),
    ],
)
# Generate embeddings for each chunk of plasmid sequence
def generate_embeddings(data):
    dna_chunk = data["chunk"]
    chunk_index = data["index"]

    # Load cached embedding model
    model = AutoModel.from_pretrained(
        CHECKPOINT, cache_dir=BEAM_VOLUME_PATH, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

    # Inference
    inputs = tokenizer(dna_chunk, return_tensors="pt")["input_ids"]
    hidden_states = model(inputs)[0]
    embedding_max = torch.max(hidden_states[0], dim=0)[0]

    # Write embeddings to a file
    file_path = f"/tmp/embedding_chunk_{chunk_index}.txt"
    with open(file_path, "w") as f:
        f.write("\n".join(map(str, embedding_max.tolist())))

    # Beam generates a pre-signed URL to access this file
    output_file = Output(path=file_path)
    output_file.save()

    return {"output_url": output_file.public_url()}


# Calculate chunk size for .map()
def chunk_sequence(sequence, chunk_size):
    return [sequence[i : i + chunk_size] for i in range(0, len(sequence), chunk_size)]


@function(
    image=Image(
        python_version="python3.11",
        python_packages=[
            "transformers",
            "sentencepiece==0.1.99",
            "accelerate==0.23.0",
            "torch",
            "biopython",
            "einops",
        ],
    ),
    # Path with the GenBank downloads
    volumes=[Volume(name="seq", mount_path="./seq")],
)
# Retrieve GenBank download, spawn containers in chunks of base pairs
def main():
    dna_sequence = read_dna_sequence(DNA_FILE_PATH)
    if dna_sequence:
        dna_chunks = chunk_sequence(dna_sequence, CHUNK_SIZE)
        chunk_data = [
            {"chunk": chunk, "index": index} for index, chunk in enumerate(dna_chunks)
        ]
        results = generate_embeddings.map(
            chunk_data
        )  # Spawn a remote container for each chunk
        for result in results:
            print(f"Embedding saved to: {result}")


if __name__ == "__main__":
    main.remote()
