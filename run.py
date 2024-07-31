#!/usr/bin/env python3
import argparse
import json
import subprocess
import re
import datetime
import os 

def main():
    ips = os.environ.get("HOSTS")
    if not ips:
        print("No hosts were defined")
        return
    
    ip_list = ips.split(',')

    parser = argparse.ArgumentParser(description='Runs mlperf on growing number of instances')
    parser.add_argument("start", type=int, help="Starting number of instances")
    parser.add_argument("--end", type=int, default=len(ip_list), help="Ending number of instances")
    parser.add_argument("--jump", type=int, default=1, help="The jump in nodes count")
    parser.add_argument("--power2", action='store_true', help="jumps in power of 2")
    parser.add_argument("--acc", type=int, default=3, help="Number of accelerators per instance")
    parser.add_argument("--readers", type=int, default=4, help="Number of readers per gpu")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for the results")
    parser.add_argument('--dryrun', action='store_true', help="dryrun flag")
    parser.add_argument('--epochs', type=int, help="amount of epochs to run for each instance")

    args = parser.parse_args()

    r = range(args.start, args.end, args.jump)
    if args.power2:
        r = []
        curr_start = 1
        while curr_start < args.end:
            if curr_start >= args.start:
                r.append(curr_start)
            curr_start *= 2
            
    for i in r:
        ips_subset = ip_list[:i]
        files = get_num_of_files(args.acc, i)
        cmd = ["./benchmark.sh", "run",
                        "--hosts", ",".join(ips_subset),
                        "--workload", "unet3d",
                        "--accelerator-type", "h100",
                        "--num-accelerators", str(args.acc * i), 
                        "--results-dir", f"{args.outdir}/{i}",
                        "--param", "dataset.data_folder=/mnt/volumez/mlperf/unet3d_data",
                        "--param", f"dataset.num_files_train={files}",
                        "--param", f"reader.read_threads={args.readers}", 
                        "--param", "dataset.num_subfolders_train=100",
                        "--param", "checkpoint.checkpoint_folder=/mnt/volumez/checkpoint"]
        
        if args.epochs:
            cmd.extend(["--param", f"train.epochs={args.epochs}"])
        print(f"\nCommand to run mlperf with {i} instances and {args.acc * i} gpus ({files} files):\n{' '.join(cmd)}")

        if args.dryrun:
            continue

        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running mlperf with {i} instances and {args.acc * i} gpus. Number of files: {files}")
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