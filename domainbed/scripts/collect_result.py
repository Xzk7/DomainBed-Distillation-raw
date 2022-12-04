# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Example usage:
python -u -m domainbed.scripts.list_top_hparams \
    --input_dir domainbed/misc/test_sweep_data --algorithm ERM \
    --dataset VLCS --test_env 0
"""

import collections


import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q, hashable
import warnings
import hashlib
import pandas as pd

def todo_rename(records, selection_method, latex):

    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (grouped_records
                    .filter_equals(
                        "dataset, algorithm, test_env",
                        (dataset, algorithm, test_env)
                    ).select("sweep_acc"))
                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--test_envs', type=int, nargs='+', required=True)
    parser.add_argument('--weight_mgds', type=float, nargs='+', required=False)
    args = parser.parse_args()

    records_raw = reporting.load_records(args.input_dir)
    print("Total records:", len(records_raw))
    envlist = list(records_raw[0].keys())
    envlist = list(filter(lambda a: "env" in a, envlist))

    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        # model_selection.LeaveOneOutSelectionMethod,
        model_selection.OracleSelectionMethod,
    ]

    for selection_method in SELECTION_METHODS:
        print(f'Model selection: {selection_method.name}')

        for test_env in args.test_envs:
            if "Distill" in args.algorithm:
                records_without_distill = reporting.load_records(args.input_dir.replace('Distill', ''))
                print("Total records:", len(records_without_distill))
                # records_without_distill = reporting.get_grouped_records(records_without_distill)
                for weight_mgd in args.weight_mgds:
                    current_envlist = ["test_env", "weight_mgd"] + envlist
                    records = reporting.get_grouped_records_Distill(records_raw)
                    records = records.filter(
                        lambda r:
                        r['dataset'] == args.dataset and
                        r['algorithm'] == args.algorithm and
                        r['test_env'] == test_env and
                        r['weight_mgd'] == weight_mgd
                    )
                    # print(len(records))
                    # records_wd = records_without_distill.filter(
                    #     lambda r:
                    #     r['dataset'] == args.dataset and
                    #     r['algorithm'] == args.algorithm.replace('Distill', '') and
                    #     r['test_env'] == test_env
                    # )
                    # print(len(records_wd))
                    for group in records:
                        print(f"trial_seed: {group['trial_seed']}")
                        best_hparams = selection_method.hparams_accs(group['records'])
                        print(best_hparams)
                        for run_acc, hparam_records in best_hparams:
                            print(f"\t{run_acc}")
                            hparams_teacher = hparam_records[0]['hparams'].copy()
                            hparams_wd = hparam_records[0]['hparams'].copy()
                            hparams_teacher['resnet18'] = False
                            records_teacher = reporting.load_records(args.input_dir.replace('Distill', '').replace('18', '50'))
                            if not 'model' in records_teacher[0]['hparams'].keys() and 'model' in hparams_teacher.keys():
                                hparams_teacher.pop("model")
                                hparams_wd.pop("model")
                            if not 'pretrained' in records_teacher[0]['hparams'].keys() and 'pretrained' in hparams_teacher.keys():
                                hparams_teacher.pop('pretrained')
                                hparams_wd.pop("pretrained")
                            records_wd = records_without_distill.filter(
                                lambda r:
                                r["args"]['dataset'] == args.dataset and
                                r["args"]['algorithm'] == args.algorithm.replace('Distill', '') and
                                r["args"]['test_envs'] == [test_env] and
                                r['hparams'] == hparams_wd
                            )
                            records_wd = reporting.get_grouped_records(records_wd)
                            records_teacher = records_teacher.filter(
                                lambda r:
                                r['args']['dataset'] == args.dataset and
                                r['args']['algorithm'] == args.algorithm.replace("Distill", "") and
                                r['args']['test_envs'] == [test_env] and
                                r['hparams'] == hparams_teacher
                            )
                            best_hparams_wd = selection_method.hparams_accs(records_wd[0]['records'])
                            chosen_record = records_teacher.sorted(lambda r: r['step'])[-1]
                            firstline = [test_env, '--']
                            secondline = [test_env, '--']
                            for run_acc_wd, hparam_records_wd in best_hparams_wd:
                                if hparam_records_wd[0]['hparams']['batch_size'] == hparam_records[0]['hparams']['batch_size']:
                                    record_run_acc_wd = run_acc_wd
                                    break

                            csv_data = [test_env, weight_mgd]
                            for env in envlist:
                                firstline.append(float('%.4f' % chosen_record[env]) * 100.)
                                if env in run_acc.keys():
                                    csv_data.append(float('%.4f' % run_acc[env])*100.)
                                    secondline.append(float('%.4f' % run_acc_wd[env])*100.)
                                elif "in" in env:
                                    csv_data.append(float('%.4f' % run_acc["test_in_acc"])*100.)
                                    secondline.append(float('%.4f' % run_acc_wd["test_in_acc"])*100.)
                                else:
                                    csv_data.append(float('%.4f' % run_acc["test_out_acc"])*100.)
                                    secondline.append(float('%.4f' % run_acc_wd["test_out_acc"])*100.)
                            hparam_hash = hashlib.md5(str(hparam_records[0]['hparams']).encode('utf-8')).hexdigest()
                            # print(test_env, weight_mgd, hparam_records[0]['hparams']["batch_size"])
                            path = os.path.join(args.work_dir, selection_method.name.split(' ')[0]+'-'+hparam_hash+".csv")
                            # print(path)
                            # print([firstline, secondline, csv_data])
                            if weight_mgd == args.weight_mgds[0]:
                                df = pd.DataFrame(data=[firstline, secondline, csv_data])
                            else:
                                df = pd.DataFrame(data=[csv_data])
                            if not os.path.exists(path):
                                df.to_csv(path, header=current_envlist, index=False, mode='a')
                            else:
                                df.to_csv(path, header=False, index=False, mode='a')
                            for r in hparam_records:
                                assert (r['hparams'] == hparam_records[0]['hparams'])
                            print("\t\thparams:")
                            for k, v in sorted(hparam_records[0]['hparams'].items()):
                                print('\t\t\t{}: {}'.format(k, v))
                            print("\t\toutput_dirs:")
                            output_dirs = hparam_records.select('args.output_dir').unique()
                            for output_dir in output_dirs:
                                print(f"\t\t\t{output_dir}")

            else:
                current_envlist = ["test_env"] + envlist
                records = reporting.get_grouped_records(records_raw)
                records = records.filter(
                    lambda r:
                    r['dataset'] == args.dataset and
                    r['algorithm'] == args.algorithm and
                    r['test_env'] == test_env
                )
                for group in records:
                    print(f"trial_seed: {group['trial_seed']}")
                    best_hparams = selection_method.hparams_accs(group['records'])
                    for run_acc, hparam_records in best_hparams:
                        print(f"\t{run_acc}")
                        csv_data = [test_env]
                        for env in envlist:
                            if env in run_acc.keys():
                                csv_data.append(float('%.4f' % run_acc[env]) * 100.)
                            elif "in" in env:
                                csv_data.append(float('%.4f' % run_acc["test_in_acc"]) * 100.)
                            else:
                                csv_data.append(float('%.4f' % run_acc["test_out_acc"]) * 100.)
                        hparam_hash = hashlib.md5(str(hparam_records[0]['hparams']).encode('utf-8')).hexdigest()
                        print(hparam_hash, hparam_records[0]['hparams'])
                        path = os.path.join(args.work_dir,
                                            selection_method.name.split(' ')[0] + '-' + hparam_hash + ".csv")
                        # 一次写入一行
                        df = pd.DataFrame(data=[csv_data])
                        if not os.path.exists(path):
                            df.to_csv(path, header=current_envlist, index=False, mode='a')
                        else:
                            df.to_csv(path, header=False, index=False, mode='a')
                        for r in hparam_records:
                            assert (r['hparams'] == hparam_records[0]['hparams'])
                        print("\t\thparams:")
                        for k, v in sorted(hparam_records[0]['hparams'].items()):
                            print('\t\t\t{}: {}'.format(k, v))
                        print("\t\toutput_dirs:")
                        output_dirs = hparam_records.select('args.output_dir').unique()
                        for output_dir in output_dirs:
                            print(f"\t\t\t{output_dir}")

