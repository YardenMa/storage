#!/usr/bin/env python3
import matplotlib.pyplot as plt


def main():
    results = {
        "no_sleep": {
            1: 97.794,
            4: 97.201,
            7: 96.923,
            10: 96.825,
            25: 96.581,
            40: 96.210,
            50: 96.058,
            60: 95.398,
            70: 94.873,
            100: 94.580,
            150: 93.346,
        },
        "with_sleep": {
            2: 96.363,
            4: 95.340,
            8: 94.290,
            12: 93.602,
            16: 92.617,
            20: 91.896,
            24: 91.663,
            28: 91.271,
            32: 90.881,
            40: 90.474,
            48: 89.859,
            56: 89.327,
            64: 89.161,
            128: 87.2969,
            150: 87.0365,
        }
    }
    for exp_name, exp_results in results.items():
        X = list(exp_results.keys())
        Y = list(exp_results.values())
        plt.plot(X, Y, label=exp_name)
        
    plt.plot(X, [90.0]*len(X), label="Threshold")
    plt.xlabel("Number of nodes")
    plt.ylabel("GPU mean utilization%")
    plt.title("GPU utilization% per number of nodes")

    plt.legend() 
    plt.show()


main()