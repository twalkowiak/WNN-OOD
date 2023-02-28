#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import random

from ood.max_softmax import MaxSoftmax
from ood.lof import LOF
from ood.mahalanobis import Mahalanobis
from ood.calculate_metrics import CalculateMetrics

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

files_to_process = []
main_folder = "../features/"
method_folders = [
    os.path.join(
        main_folder,
        o) for o in os.listdir(main_folder) if os.path.isdir(
            os.path.join(
                main_folder,
                o))]

for method_folder in method_folders:
    models_folders = [
        os.path.join(
            method_folder,
            o) for o in os.listdir(method_folder) if os.path.isdir(
            os.path.join(
                method_folder,
                o))]

    for model_folder in models_folders:
        files = [f for f in os.listdir(model_folder) if os.path.isfile(
            os.path.join(model_folder, f)) and ".pickle" in f]
        for f in files:
            if "train" in f:
                train_file = f

            if "test" in f:
                test_file = f

        files.remove(train_file)
        files.remove(test_file)
        files_to_process.append((model_folder, train_file, test_file, files))

for files in files_to_process:
    print("\n", files)


def parse_ood_result(results):
    tnr = int(results['tnr_95'] * 10000) / 100
    auc = int(results['auc'] * 10000) / 100
    acc = int(results['max_bin_acc'] * 10000) / 100
    _aupr = (results['aupr_in'] + results['aupr_out']) / 2
    aupr = int(_aupr * 10000) / 100

    return tnr, auc, acc, aupr


def run(white_list, if_attacks=False):
    oods = [MaxSoftmax(), LOF(LOF.EUCLIDEAN), Mahalanobis()]
    for path, train_file, test_file, rest_files in files_to_process:
        print(path)

        model_name = path.split("/")[3]
        extraction_name = path.split("/")[2]

        train_df = pd.read_pickle("{}/{}".format(path, train_file))
        if not if_attacks:
            test_df = pd.read_pickle("{}/{}".format(path, test_file))
        else:
            test_df = pd.read_pickle(
                "{}/{}".format(path, test_file)).sample(n=1000, random_state=0)

        for ood in oods:
            any_to_run = False
            for file in rest_files:
                name = file.split(".")[0]
                if not os.path.isfile(
                        "{}/results/{}_{}.pickle".format(path, ood.name, name)):
                    any_to_run = True

            print("\t >", ood.name)
            if not any_to_run:
                print("\t\t > all was calculated")
                continue

            if not if_attacks:
                score_path = "{}/scores/{}_{}.npy".format(
                    path, ood.name, "test")
            else:
                score_path = "{}/scores/{}_{}.npy".format(
                    path, ood.name, "test_attacks")
            directory = os.path.dirname(score_path)
            Path(directory).mkdir(parents=True, exist_ok=True)

            ood.clear()
            ood.fit(train_df)
            ood.known_out = ood.test(test_df)
            np.save(score_path, ood.known_out)

            for file in rest_files:
                name = file.split(".")[0]
                if name not in white_list:
                    continue
                print("\t\t >", name)

                result_path = "{}/results/{}_{}.pickle".format(
                    path, ood.name, name)
                score_path = "{}/scores/{}_{}.npy".format(path, ood.name, name)
                if os.path.isfile(result_path):
                    print("\t\t\t > it was calculated")
                    continue

                df = pd.read_pickle("{}/{}".format(path, file))
                # if len(df.index) != len(test_df.index):
                #    print("\t\t\t > len(unknown) != len(test)")
                #    continue

                directory = os.path.dirname(result_path)
                Path(directory).mkdir(parents=True, exist_ok=True)
                directory = os.path.dirname(score_path)
                Path(directory).mkdir(parents=True, exist_ok=True)

                ood.unknown_out = ood.test(df)
                np.save(score_path, ood.unknown_out)

                results = CalculateMetrics().run(ood)
                tnr, auc, acc, aupr = parse_ood_result(results)
                row = {
                    'model': [model_name], 'ood_method': [ood.name], 'unknown_dataset': [name],
                    'tnr_95': [tnr], 'auc': [auc], 'acc': [acc], 'aupr': [aupr], 'extraction': extraction_name
                }
                print("\t\t\t", {'tnr_95': [tnr], 'auc': [
                      auc], 'acc': [acc], 'aupr': [aupr]})

                df_results = pd.DataFrame.from_dict(row)
                df_results.to_pickle(result_path)


run(["noise", "svhn", "cifar100"], False)
run(["CW", "DeepFool", "FGSM", "OnePixel", "PGD", "Square", "SparseFool"], True)
