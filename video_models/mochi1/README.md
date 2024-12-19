# Mochi-1 API Example

A text-to-video API service running Mochi-1

## Deployment

```bash
beam deploy app.py:generate_video
```

## API Usage
Send a POST request with:
```bash
{
    "prompt": "Your prompt for video",
}
```

### Example Request:

```json
{
    "prompt": "The camera follows behind a rugged green Jeep with a black snorkel as it speeds along a narrow dirt trail cutting through a dense jungle. Thick vines hang from towering trees with sprawling canopies, their leaves forming a vibrant green tunnel above the vehicle. Mud splashes up from the Jeep’s tires as it powers through a shallow stream crossing the path. Sunlight filters through gaps in the trees, casting dappled golden light over the scene. The dirt trail twists sharply into the distance, overgrown with wild ferns and tropical plants. The vehicle is seen from the rear, leaning into the curve as it maneuvers through the untamed terrain, emphasizing the adventure of the rugged journey. The surrounding jungle is alive with texture and color, with distant mountains barely visible through the mist and an overcast sky heavy with the promise of rain.",
}
```

### Example Response:

A generated video file will be returned. 

## Video Example

Here’s an example of the generated video output:

<video controls width="640" height="360">
  <source src="https://app.beam.cloud/output/id/dc443a80-7fcc-42bc-928b-4605e41b0825" type="video/mp4">
</video>
