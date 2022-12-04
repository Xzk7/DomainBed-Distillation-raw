# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os

import tqdm

from domainbed.lib.query import Q

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]:
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env)
            result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()])

def get_grouped_records_Distill(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]:
            if "weight_wsld" in r["args"].keys() and "distillation_type" in r["args"].keys():
                group = (r["args"]["trial_seed"],
                         r["args"]["dataset"],
                         r["args"]["algorithm"],
                         test_env,
                         r["args"]["weight_mgd"],
                         r["args"]["weight_wsld"],
                         r["args"]["distillation_type"],
                         r["args"]["seed"],)
                result[group].append(r)
            else:
                group = (r["args"]["trial_seed"],
                         r["args"]["dataset"],
                         r["args"]["algorithm"],
                         test_env,
                         None,
                         None,
                         r["args"]["weight_mgd"],
                         r["args"]["seed"],)
                result[group].append(r)

    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e, "weight_mgd": w1, "weight_wsld": w2,
                   "distillation_type": type, "seed": s,
                   "records": Q(r)} for (t, d, a, e, w1, w2, type, s), r in result.items()])

