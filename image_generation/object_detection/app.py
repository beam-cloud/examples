from beam import Image, endpoint, Output, env
import io
import base64

# Since these packages are only installed remotely on Beam, this block ensures the interpreter doesn't try to import them locally
if env.is_remote():
    from torchvision import models, transforms
    from PIL import Image as PILImage
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import torch
    from torchvision import models, transforms

    # Define the image transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


# The beam container image that this code will run on
image = Image(python_version="python3.9").add_python_packages(
    [
        "torch",
        "torchvision",
        "pillow",
        "matplotlib",
    ]
)


# Pre-load models onto the container
def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


def detect_objects(model, image):
    image_tensor = transform(image).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the bounding boxes and labels
    boxes = predictions[0]["boxes"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()

    # Filter out low-confidence detections
    threshold = 0.5
    boxes = boxes[scores >= threshold]
    labels = labels[scores >= threshold]

    return boxes, labels


def visualize_detection(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw the bounding boxes
    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return PILImage.open(buf)


@endpoint(
    image=image,
    on_start=load_model,
    keep_warm_seconds=60,
    cpu=2,
    gpu="A10G",
    memory="16Gi",
)
def predict(context, image_base64: str):
    # Retrieve pre-loaded model from loader
    model = context.on_start_value

    # Decode the base64 image
    image_data = base64.b64decode(image_base64)
    image = PILImage.open(io.BytesIO(image_data))

    # Perform object detection
    boxes, labels = detect_objects(model, image)

    # Visualize the results
    result_image = visualize_detection(image, boxes)

    # Save image file
    output = Output.from_pil_image(result_image).save()

    # Retrieve pre-signed URL for output file
    url = output.public_url()

    return {"image": url, "boxes": boxes.tolist()}
