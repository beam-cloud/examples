"""
GPU_TYPE=T4 locust -f locustfile.py --host=https://your-endpoint --headless --run-time 2m --users 1 --spawn-rate 1
"""
from locust import HttpUser, task, events, constant
import csv
import time
import os


class SDXLUser(HttpUser):
    wait_time = constant(1)

    @task
    def test_sdxl_inference(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer API_TOKEN",
        }

        payload = {
            "prompt": "A happy little tree in the middle of a forest",
        }

        start_time = time.time()
        response = self.client.post("/", json=payload, headers=headers)
        end_time = time.time()

        total_request_time = (end_time - start_time) * 1000  # Convert to ms
        json_response = response.json()
        inference_time = json_response.get("inference_time_ms")

        events.request.fire(
            request_type="POST",
            name="sdxl_inference",
            response_time=total_request_time,
            response_length=len(response.content),
            exception=None,
        )

        # Save inference time
        if inference_time is not None:
            with open("inference_times.csv", "a") as f:
                f.write(f"{inference_time}\n")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    with open("inference_times.csv", "w") as f:
        f.write("")

    print("GPU Benchmark starting...")


@events.quitting.add_listener
def save_results(environment, **kwargs):
    inference_times = []

    try:
        with open("inference_times.csv", "r") as f:
            for line in f.readlines():
                if line.strip():
                    inference_times.append(float(line.strip()))
    except FileNotFoundError:
        print("No inference times recorded")
        return

    if not inference_times:
        print("No valid inference times recorded")
        return

    avg_inference_time = sum(inference_times) / len(inference_times)
    rps = 1000 / avg_inference_time if avg_inference_time > 0 else 0

    gpu_type = os.environ.get("GPU_TYPE", "A10G")  

    # Save results
    filename = f"{gpu_type.lower()}_benchmark.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["GPU", "Avg Inference Time (ms)", "Requests Per Second"])
        writer.writerow([gpu_type, avg_inference_time, rps])

    print(f"\n===== BENCHMARK RESULTS =====")
    print(f"GPU: {gpu_type}")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Requests Per Second: {rps:.4f}")
    print(f"Results saved to {filename}")
