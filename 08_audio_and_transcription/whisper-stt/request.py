"""
*** Whisper Benchmarking Script ***

This script is used to benchmark the inference time and cold boot.

Requests are sent in batches. In between batches, we use the Beam CLI to kill the container 
that is running in order to demonstrate the cold boot latency (`beam container list` and `beam container stop`)
"""

import requests
import time
import subprocess

BEAM_AUTH_TOKEN = ""  # Add your Beam Auth Token, you can find it in the dashboard by clicking the 'Call API' button on your app

url = "https://app.beam.cloud/endpoint/whisper/v1"
headers = {
    "Connection": "keep-alive",
    "Authorization": f"Bearer {BEAM_AUTH_TOKEN}",
    "Content-Type": "application/json",
}
data = {
    # Any arbitrary Youtube URL, we use a short 10s video for the sake of this example
    "video_url": "https://www.youtube.com/watch?v=YVlCtJ52Juc&ab_channel=minutephysics"
}

times = []
cold_boot_times = []
batch_size = 5
total_requests = 50
wait_time = 20

for batch in range(total_requests // batch_size):
    # Measure the time for the first request after the waiting period
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    end_time = time.time()

    elapsed_time = end_time - start_time
    cold_boot_times.append(elapsed_time)
    times.append(elapsed_time)

    print(
        f"Batch {batch+1}, Request 1 (Cold Boot): Status Code: {response.status_code}, Time: {elapsed_time:.4f} seconds"
    )

    # Measure the time for the remaining requests in the batch
    for i in range(1, batch_size):
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        print(
            f"Batch {batch+1}, Request {i+1}: Status Code: {response.status_code}, Time: {elapsed_time:.4f} seconds"
        )

    # Run "beam container list" to show all running containers
    result = subprocess.run(
        ["beam", "container", "list"], capture_output=True, text=True
    )
    print(result.stdout)

    # Run "beam container stop" for each container to demonstrate cold boot latency
    lines = result.stdout.split("\n")
    container_ids = []
    for line in lines:
        if "RUNNING" in line:
            container_id = line.split()[0]
            container_ids.append(container_id)

    for container_id in container_ids:
        subprocess.run(["beam", "container", "stop", container_id])

    if batch < (total_requests // batch_size) - 1:
        print(f"Waiting for {wait_time} seconds before the next batch...")
        time.sleep(wait_time)

# Benchmark report
total_time = sum(times)
average_time = total_time / len(times)
max_time = max(times)
min_time = min(times)

# Cold boot report
cold_boot_total_time = sum(cold_boot_times)
cold_boot_average_time = cold_boot_total_time / len(cold_boot_times)

print("\nBenchmark Report")
print(f"Total Requests: {len(times)}")
print(f"Total Time: {total_time:.4f} seconds")
print(f"Average Time per Request: {average_time:.4f} seconds")
print(f"Maximum Time for a Request: {max_time:.4f} seconds")
print(f"Minimum Time for a Request: {min_time:.4f} seconds")

print("\nCold Boot Report")
print(f"Total Cold Boot Requests: {len(cold_boot_times)}")
print(f"Total Cold Boot Time: {cold_boot_total_time:.4f} seconds")
print(f"Average Cold Boot Time per Request: {cold_boot_average_time:.4f} seconds")
