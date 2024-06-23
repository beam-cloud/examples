"""
Saving Outputs.

The code below shows how to retrieve a pre-signed URL to a file saved during function execution.
"""

from beam import Image as BeamImage, Output, function


@function(
    image=BeamImage(
        python_packages=[
            "pillow",
        ],
    ),
)
def save_image():
    from PIL import Image as PILImage

    # Example PIL image
    pil_image = PILImage.new(
        "RGB", (100, 100), color="white"
    )  # Creating a 100x100 white image

    # Save image file
    output = Output.from_pil_image(pil_image)
    output.save()

    # Retrieve pre-signed URL for output file
    url = output.public_url(expires=400)
    print(url)

    # Print other details about the output
    print(f"Output ID: {output.id}")
    print(f"Output Path: {output.path}")
    print(f"Output Stats: {output.stat()}")
    print(f"Output Exists: {output.exists()}")

    return {"image": url}


if __name__ == "__main__":
    save_image()
