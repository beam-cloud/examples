"""
** Mounting Volumes ** 

Beam allows you to create highly-available storage volumes that can be used across tasks. 
You might use volumes for things like storing model weights or large datasets.
"""

from beam import function, Volume


VOLUME_PATH = "./model_weights"


@function(
    volumes=[Volume(name="model-weights", mount_path=VOLUME_PATH)],
)
def load_model():
    from transformers import AutoModel

    # Load model from cloud storage cache
    AutoModel.from_pretrained(VOLUME_PATH)
