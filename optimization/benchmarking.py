import time

import torch


def benchmark_inference(model, image_path, question, n_runs=10):
    timings = []

    for i in range(n_runs):
        start = time.time()
        response = model.chat(image_path, question)
        end = time.time()
        timings.append(end - start)

        if i == 0:
            print(f"Sample response: {response[:100]}...")

    avg_time = sum(timings) / len(timings)
    std_time = torch.tensor(timings).std().item()

    print(f"\nResults ({n_runs} runs):")
    print(f"Average latency: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Min: {min(timings):.3f}s, Max: {max(timings):.3f}s")

    return timings
