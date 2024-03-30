#!/usr/bin/env python
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from source.evaluation import Evaluation

default_dataset_path = 'vost'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='Path to the dataset folder containing the JPEGImages, Annotations, '
                                                   'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_dataset_path)
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    required=True)
parser.add_argument('--re', action='store_true')
args, _ = parser.parse_known_args()
dataset_path_dict = {
    'vost': '../aot_plus/datasets/VOST',
    'long_videos': '../aot_plus/datasets/long_videos',
}
args.dataset_path = dataset_path_dict[args.dataset_path]
csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

print(f"Evaluating {args.results_path}")

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)
if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path) and not args.re:
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences ...')
    # Create dataset and evaluate
    dataset_eval = Evaluation(dataset_root=args.dataset_path, gt_set=args.set)
    metrics_res = dataset_eval.evaluate(args.results_path)
    J = metrics_res['J']
    J_last = None
    if 'J_last' in metrics_res:
        J_last = metrics_res['J_last']

    # Generate dataframe for the general results
    g_measures = ['J-Mean', 'J-Recall', 'J-Decay', 'J_last-Mean', 'J_last-Recall', 'J_last-Decay']
    g_res = np.array([np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(J_last["M"]), np.mean(J_last["R"]), np.mean(J_last["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.6f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'J_last-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    J_last_per_object = [J_last['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, J_last_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.6f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
