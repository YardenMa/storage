#!/usr/bin/env python3
import argparse
import json

from pathlib import Path
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plots mlperf results over growing number of instances')
    parser.add_argument("results_dir", type=str, help="Path to results dir")
    args = parser.parse_args()

    results_dir = args.results_dir

    pathlist = Path(results_dir).glob('*/summary.json')
    metrics_per_n_hosts = dict()

    for path in pathlist:
        summary_file = str(path)
        n_hosts = int(path.parent.name)

        with open(summary_file, 'r') as f:
            summary = json.load(f)
            gpu_util_mean = summary["metric"]["train_au_mean_percentage"]
            gpu_util_std = summary["metric"]["train_au_stdev_percentage"]

        metrics_per_n_hosts[n_hosts] = {
            "gpu_util_mean": gpu_util_mean,
            "gpu_util_std": gpu_util_std,
        }

    metrics_per_n_hosts = dict(sorted(metrics_per_n_hosts.items()))

    X = list(metrics_per_n_hosts.keys())
    Y = [v["gpu_util_mean"] for v in metrics_per_n_hosts.values()]
    max_std = max(v["gpu_util_std"] for v in metrics_per_n_hosts.values())
    print(f"Max std is: {max_std:.3f}")

    plt.plot(X, Y, label="Actual")
    plt.plot(X, [90.0]*len(X), label="Threshold")

    plt.xlabel("Number of nodes")
    plt.ylabel("GPU mean utilization%")
    plt.title("GPU utilization% per number of nodes")

    plt.legend() 
    plt.show()


main()