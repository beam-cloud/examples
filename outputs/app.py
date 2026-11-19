from beam import Image, Output, function

@function(app="examples", image=Image().add_python_packages(["pillow"]))
def save_image():
    from PIL import Image as PILImage

    # Generate PIL image
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
    print(f"Output Exists: {output.exists()}")

    return {"image": url}


if __name__ == "__main__":
    save_image.remote()
