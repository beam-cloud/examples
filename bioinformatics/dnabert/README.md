# Serverless Bioinformatics

> TL;DR: [Beam](https://beam.cloud) is a platform that makes it easy to run serverless cloud functions with simple Python decorators.

## The Problem

No one loves using AWS.

AWS is extremely powerful, modular, and complicated.

If you've worked in software for a few years, you're probably pretty good at AWS by now. But it still involves _so many_ unnecessary steps.

- You need to think about IAM roles.
- You need to manage a Kubernetes cluster (wait, what's a pod?)
- You need to think about security groups.

It's not impossible to learn it, of course. Before IDEs, we programmed on punch cards and it worked pretty well. But IDEs made us faster and more productive.

Today, the cloud feels a bit like programming with punch cards. And we think there's a better way.

## Python Decorators and CLI Commands

Beam is a new cloud platform where you can add simple Python decorators to your code to instantly run functions on the cloud.

Suppose you want to batch process data from GenBank. You need a containerized cloud function:

```python
from beam import function, Image

@function(image=Image(python_packages=["biopython"]))
def download_files():
    ...
```

By adding this `@function` decorator, this code will run on the cloud instead of your laptop.

Behind the scenes, here's what Beam is doing:

- Creating a Runc container with your image
- Scheduling your container on a server
- Running your code and streaming the logs back to your shell

You don't need Docker installed and you don't need an AWS account.

It's pretty cool!

```sh
$ python download-dna.py # This is running on the cloud, not your laptop

=> Running function: <download-dna:download>
=> Function complete <8bb1e80b-533c-4f91-8eba-e8a6f899ed7c>
```

Of course, you could also run this locally. But what if you wanted a GPU attached? Perhaps you'd like to add a custom base image as well?

Let's add `gpu` and `base_image` parameters.

```python
from beam import function, Image

@function(
    gpu="A100-40",
    image=Image(
        python_packages=["biopython"],
        base_image="nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04",
    ),
)
def download_files():
    ...
```

Now your function will run on a GPU.

Of course, your laptop might already have a GPU. But what if you wanted to batch process data on 100 containers in parallel?

```python
from beam import function, Image


@function(gpu="A100-40", image=Image(python_packages=["biopython"]))
def download_files():
    ...

def main():
    for i in range(0, 100):
        download_files.map(i) # Spin up 100 containers in the cloud
```

Isn't this just a wrapper on AWS? Well, AWS sells servers, and this service runs your code on servers, so sort of.

But that's the extent of this analogy.

Behind the scenes, Beam leverages a network of servers across the world and is powered by a custom container runtime, scheduler, and image cache. The code is fully open-source, [and you can see it here](https://github.com/beam-cloud/beta9).

# Plasmid Analysis: An end-to-end example for bioinformatics

In this example, we'll download plasmid data from GenBank and generate ML embeddings. To speed it up, we'll shard the ML embedding process across many containers in parallel.

## Downloading Data & Filesystem I/O

The first step is downloading plasmid data from GenBank.

```python

from beam import function, Image, Volume, env
import os

if env.is_remote():
    from Bio import Entrez, SeqIO

image = Image(python_packages=["biopython"])
BEAM_VOLUME_PATH = "./seq"


@function(volumes=[Volume(name="seq", mount_path=BEAM_VOLUME_PATH)], image=image)
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
    download.remote("CM004190.1")
```

We'll run this like any ordinary Python file. But because we've got that `.remote()` method, this code will run on the cloud instead of our laptop.

```sh
$ python download-dna.py

=> Running function: <download-dna:download>
Sequence ID: CM004190.1
Length: 33445071
Description: Pan troglodytes isolate Yerkes chimp pedigree #C0471 (Clint) chromosome 21, whole genome shotgun
sequence
=> Function complete <8bb1e80b-533c-4f91-8eba-e8a6f899ed7c>
```

We can verify that the GenBank data was downloaded by using the CLI to view the Volumes:

```sh
$ beam ls seq

  Name                           Size   Modified Time   IsDir
 ─────────────────────────────────────────────────────────────
  CM004190.1.gb              7.92 KiB   5 minutes ago   No
```

## Running Parallel Jobs

The next step is to send the plasmid sequence to an ML model to generate embeddings.

Sequences are embedded in batches of 3000 base pairs. We use the `.map()` method in Beam, which spawns an individual container for each batch of sequences.

```python
from beam import function, Image, Volume, env, Output

# This packages will only get imported in the remote cloud runtime
if env.is_remote():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from Bio import SeqIO


CHECKPOINT = "RaphaelMourad/Mistral-DNA-v1-422M-hg38"  # Embedding model
BEAM_VOLUME_PATH = "./cached_models"  # Model is cached here
DNA_FILE_PATH = "./seq/AE017046.1.gb"  # https://www.ncbi.nlm.nih.gov/nuccore/AE017046
CHUNK_SIZE = 3000  # Run in batches of 3000 base pairs each


def read_dna_sequence(file_path):
    records = list(SeqIO.parse(file_path, "genbank"))
    return "".join([str(record.seq) for record in records if record.seq])


@function(
    secrets=["HF_TOKEN"],  # Hugging Face API key for embedding model
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
        ],
    ),
    volumes=[
        Volume(
            name="cached_models", mount_path=BEAM_VOLUME_PATH
        ),  # Embedding model is cached here
        Volume(name="seq", mount_path="./seq"),  # Path with the GenBank downloads
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
    volumes=[Volume(name="seq", mount_path="./seq")],  # Path with the GenBank downloads
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
```

We'll run this like an ordinary Python function:

```sh
$ python app.py

=> Running function: <app:main>
=> Building image
=> Using cached image
=> Syncing files
=> Files synced
=> Running function: <app:generate_embeddings>
=> Running function: <app:generate_embeddings>
=> Running function: <app:generate_embeddings>
=> Running function: <app:generate_embeddings>
=> Function complete <4c4ea387-f8cf-499e-9470-aba852f9a6c3>
Embedding saved to: {'output_url': 'https://app.beam.cloud/output/id/53f03dfe-6564-4f89-8657-888dd01ceb62'}
=> Function complete <b38ca7f5-462d-4b96-a244-fc2ffb6ef2ad>
Embedding saved to: {'output_url': 'https://app.beam.cloud/output/id/9a74a538-004d-4203-a09f-21dd9c488b67'}
=> Function complete <f962cbe9-966f-42e3-9f6b-e609996984b4>
Embedding saved to: {'output_url': 'https://app.beam.cloud/output/id/1c62ebce-6a65-429e-b32d-07bc6fe6b2bd'}
=> Function complete <3a988ff8-1fa0-4804-898a-1e4dc3d30c46>
Embedding saved to: {'output_url': 'https://app.beam.cloud/output/id/881afdc9-e13a-4aa7-8599-b41be5f60f80'}
=> Function complete <436eadd4-b5d4-45af-8f33-a00c8f3e3b77>
```

Each `output_url` can be opened in a browser to view the generated embeddings.

## Deploying Sharable Web Endpoints

You might want to deploy this code as a web API instead.

```python
from beam import function, Image

@function(gpu="A100-40", image=Image(python_packages=["biopython"]))
def download_files():
    ...
```

In your shell, run:

```
beam deploy app.py:main
```

## Limitations for Bioinformatics

Of course, there are existing, powerful platforms that many know and love. While we believe in the power of Beam, there are several things that it _cannot_ do.

- Declarative DSL -- Beam is just Python. Unlike Nextflow, there's no DSL to define a process. The Python code is the source of truth.
- File-based workflow execution -- while you _can_ read and write data to a cloud filesystem on Beam, it doesn't offer the ability to spawn tasks based on particular filenames like you can do on [Nextflow](https://nextflow-io.github.io/patterns/process-per-file-path/).
- No DAG viewer -- there's no DAG visualization tool, so you can't easily see how your application is chained together without reading the Python code.

We are not bioinformatics experts, and there are likely other capabilities missing that we aren't aware of. If we missed something important, let us know! We are eager to learn.

## Summary

In this guide, we demonstrated several cloud capabilities, created exclusively through Python decorators.

But most importantly, there are several things that weren't included:

- Writing Dockerfiles or other DSLs
- Configuring Kubernetes
- Creating EC2 resources and VPCs
- Setting permissions and access policies

That's because all of this stuff is hidden away.

Our guiding belief is that, as a scientist, you don't care about this DevOps nonsense. You're paid to discover cool and important things, not to fiddle with Kubernetes or AWS.

With Beam, you don't have to think about that stuff. Just write your Python, add some special cloud decorators, and run it on the cloud.

In our view, this is how the cloud is supposed to feel. Powerful, accessible, and invisible.
