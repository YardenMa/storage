#!/usr/bin/env python3
import argparse
import json
import subprocess
import re
import time
import datetime
import os 

def main():
    ips = "10.0.96.71,10.0.103.7,10.0.99.242,10.0.96.150,10.0.105.59,10.0.107.9,10.0.97.124,10.0.110.48,10.0.97.213,10.0.97.78,10.0.107.127,10.0.106.66,10.0.110.129,10.0.102.209,10.0.98.57,10.0.111.231,10.0.98.253,10.0.107.212,10.0.104.163,10.0.99.137,10.0.102.138,10.0.111.57,10.0.106.77,10.0.106.200,10.0.96.136,10.0.103.251,10.0.96.30,10.0.96.194,10.0.103.9,10.0.102.249,10.0.98.215,10.0.103.232,10.0.105.243,10.0.101.237,10.0.99.96,10.0.110.246,10.0.107.152,10.0.108.112,10.0.99.243,10.0.98.85,10.0.99.108,10.0.107.215,10.0.97.186,10.0.110.199,10.0.105.117,10.0.107.231,10.0.102.88,10.0.97.6,10.0.98.229,10.0.100.121,10.0.104.135,10.0.107.243,10.0.110.123,10.0.98.33,10.0.100.165,10.0.98.3,10.0.109.232,10.0.105.108,10.0.103.112,10.0.97.142,10.0.107.172,10.0.104.148,10.0.109.172,10.0.102.52,10.0.96.230,10.0.106.235,10.0.108.58,10.0.100.209,10.0.99.32,10.0.97.176,10.0.96.7,10.0.97.15,10.0.99.187,10.0.99.241,10.0.111.152,10.0.109.63,10.0.102.179,10.0.100.214,10.0.105.45,10.0.99.9,10.0.97.123,10.0.106.45,10.0.111.84,10.0.105.181,10.0.99.138,10.0.105.134,10.0.111.75,10.0.96.233,10.0.97.76,10.0.102.100,10.0.98.53,10.0.111.117,10.0.102.94,10.0.101.243,10.0.107.227,10.0.99.193,10.0.106.41,10.0.107.125,10.0.102.154,10.0.96.253,10.0.100.193,10.0.103.41,10.0.104.246,10.0.99.23,10.0.98.127,10.0.111.37,10.0.107.102,10.0.97.47,10.0.97.61,10.0.97.128,10.0.109.243,10.0.109.158,10.0.97.149,10.0.111.88,10.0.96.128,10.0.104.152,10.0.101.149,10.0.107.139,10.0.109.111,10.0.110.39,10.0.106.47,10.0.105.208,10.0.105.245,10.0.104.10,10.0.100.98,10.0.100.136,10.0.100.158,10.0.101.221,10.0.102.143,10.0.104.196,10.0.101.82,10.0.101.12,10.0.105.78,10.0.109.113,10.0.106.39,10.0.110.14,10.0.106.64,10.0.109.103,10.0.107.45,10.0.111.251,10.0.102.26,10.0.107.168,10.0.96.60,10.0.97.158,10.0.107.137,10.0.109.59,10.0.106.119,10.0.101.86"
    ip_list = ips.split(',')

    parser = argparse.ArgumentParser(description='Runs mlperf on growing number of instances')
    parser.add_argument("start", type=int, help="Starting number of instances")
    parser.add_argument("--end", type=int, default=len(ip_list), help="Ending number of instances")
    parser.add_argument("--jump", type=int, default=1, help="The jump in nodes count")
    parser.add_argument("--acc", type=int, default=3, help="Number of accelerators per instance")
    parser.add_argument("--readers", type=int, default=4, help="Number of readers per gpu")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for the results")
    args = parser.parse_args()


    for i in range(args.start, args.end, args.jump):
        ips = ip_list[:i]
        files = get_num_of_files(args.acc, i)

        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running mlperf with {i} instances and {args.acc * i} gpus. Number of files: {files}")
        
        cmd = ["./benchmark.sh", "run",
                        "--hosts", ",".join(ips),
                        "--workload", "unet3d",
                        "--accelerator-type", "h100",
                        "--num-accelerators", str(args.acc * i), 
                        "--results-dir", f"{args.outdir}/{i}",
                        "--param", "dataset.data_folder=/mnt/volumez/mlperf/unet3d_data",
                        "--param", f"dataset.num_files_train={files}",
                        "--param", f"reader.read_threads={args.readers}", 
                        "--param", "dataset.num_subfolders_train=100",
                        "--param", "checkpoint.checkpoint_folder=/mnt/volumez/checkpoint"]
        print(f"Running command: {' '.join(cmd)}")
        out = subprocess.run(cmd, stderr=subprocess.DEVNULL)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] mlperf with {i} instances and {args.acc*i} gpus is done")
        
        with open(os.path.join(args.outdir, str(i), "summary.json"), 'r') as f:
            summary = json.load(f)
            gpu_util = summary["metric"]["train_au_mean_percentage"]
            gpu_std = summary["metric"]["train_au_stdev_percentage"]
            samples_avg = summary["metric"]["train_throughput_mean_samples_per_second"]
            samples_std = summary["metric"]["train_throughput_stdev_samples_per_second"]
            bw_avg = summary["metric"]["train_io_mean_MB_per_second"]
            bw_std = summary["metric"]["train_io_stdev_MB_per_second"]

            print(f"GPU Util: {gpu_util} [{gpu_std}]")
            print(f"Samples/s: {samples_avg} [{samples_std}]")
            print(f"MB/s: {bw_avg} [{bw_std}]")

def get_num_of_files(acc, nodes):
    out = subprocess.run(["./benchmark.sh", "datasize", "--workload", "unet3d", "--accelerator-type", "h100", "--num-accelerators", str(acc*nodes), "--num-client-hosts", str(nodes), "--client-host-memory-in-gb", "9"], stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)    
    output = out.stdout.decode("utf-8")
    pattern = r"dataset\.num_files_train=(\d+)"
    match = re.search(pattern, output)
    if not match:
        raise ValueError("Number of files not found in the output")
    num_files_train = int(match.group(1))
    return num_files_train

main()