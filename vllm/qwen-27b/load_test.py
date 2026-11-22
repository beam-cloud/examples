"""
Load test for the Qwen 27B deployment.

Sends REQUESTS chat completions with at most CONCURRENCY in flight, streams
each one, and reports errors, latency, time to first token and throughput.

    pip install openai
    export QWEN_URL=https://<your-deployment>-8000.app.beam.cloud
    export QWEN_KEY=<the VLLM_API_KEY value>
    python load_test.py --requests 200 --concurrency 100
"""

import argparse
import asyncio
import os
import statistics
import time

from openai import AsyncOpenAI

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write an outline for a 10-slide deck introducing a new budgeting app.",
    "Summarize the causes of the 2008 financial crisis in five bullet points.",
    "Write a SQL query that returns the top 3 customers by revenue per region.",
    "Draft a polite email declining a meeting and proposing two other times.",
    "Compare REST and gRPC for an internal microservice API.",
    "Give me a week of vegetarian dinner ideas with a shopping list.",
    "Explain the difference between a process and a thread to a new engineer.",
]


def percentile(values, pct):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct / 100))]


async def one_request(client, args, index, limiter):
    async with limiter:
        started = time.perf_counter()
        first_token_at = None
        output_tokens = 0
        try:
            stream = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
                max_tokens=args.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking}},
            )
            async for chunk in stream:
                if chunk.choices and first_token_at is None:
                    first_token_at = time.perf_counter()
                if chunk.usage:
                    output_tokens = chunk.usage.completion_tokens
        except Exception as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"[:200]}

        finished = time.perf_counter()
        return {
            "ok": True,
            "latency": finished - started,
            "ttft": (first_token_at or finished) - started,
            "output_tokens": output_tokens,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default="qwen-27b")
    parser.add_argument("--thinking", action="store_true", help="leave thinking mode on")
    args = parser.parse_args()

    client = AsyncOpenAI(
        base_url=os.environ["QWEN_URL"].rstrip("/") + "/v1",
        api_key=os.environ["QWEN_KEY"],
        timeout=600,
        max_retries=0,
    )

    limiter = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(one_request(client, args, i, limiter) for i in range(args.requests))
    )
    elapsed = time.perf_counter() - started

    succeeded = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"requests      {len(results)} at concurrency {args.concurrency}")
    print(f"succeeded     {len(succeeded)}")
    print(f"failed        {len(failed)}")
    for error in sorted({r["error"] for r in failed})[:5]:
        print(f"  {error}")
    if not succeeded:
        return

    latencies = [r["latency"] for r in succeeded]
    ttfts = [r["ttft"] for r in succeeded]
    output_tokens = sum(r["output_tokens"] for r in succeeded)
    print(f"wall time     {elapsed:.1f}s")
    print(f"throughput    {output_tokens / elapsed:,.0f} output tokens/s, "
          f"{len(succeeded) / elapsed:.2f} requests/s")
    print(f"output tokens {output_tokens:,} total, "
          f"{output_tokens / len(succeeded):.0f} per request")
    print(f"latency       p50 {statistics.median(latencies):.1f}s, "
          f"p95 {percentile(latencies, 95):.1f}s, max {max(latencies):.1f}s")
    print(f"first token   p50 {statistics.median(ttfts):.2f}s, "
          f"p95 {percentile(ttfts, 95):.2f}s, max {max(ttfts):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
