# -*- coding: utf-8 -*-
# pylint: disable=W0108

import multiprocessing
import os
import threading
from pathlib import Path

import commentjson
import numpy
from astropy.io import fits
from joblib import Parallel, delayed, parallel_backend


def get_folders(name_folder: str, name_database: str, directories):
    folder_databases = search_with_timeout(name_folder, directories)
    subdirs = [name_database, "PLOTS", "MODELS"]
    folders = []
    for subdir in subdirs:
        folder_path = folder_databases / subdir
        folder_path.mkdir(parents=True, exist_ok=True)
        folders.append(folder_path)

    return tuple(folders)


def get_user_data_network():
    params = get_json_file_parameters()["network"]

    return params["batch_size"], params["num_epochs"], params["early_stopping"], params["type_optimizer"], params["momentum"]


def get_user_data_general():
    params = get_json_file_parameters()["general"]
    return params["NAME_DB"], params["path_folders"], params["directories"], params["percentage"], params["normalisation"]


def check_slurm_mode():
    if "SLURM_JOB_ID" in os.environ:
        if os.environ.get("SLURM_JOB_NAME") != "bash":
            print("Running in sbatch mode")
            return True

    print("Running in interactive mode")
    return False


def search_with_timeout(name: str, directories):
    condition = threading.Condition()
    found_path = []
    timeout = 5

    for directory in directories:
        searcher = threading.Thread(target=find_directory, args=(name, directory, condition, found_path))
        searcher.start()

        with condition:
            notified = condition.wait(timeout)

        if notified and found_path:
            return found_path[0]
        if notified:
            continue
        print(f"Timeout while searching in {directory}, moving to next directory.")

    return None


def get_statistics(file_paths, fn_load=fits.getdata, on_the_fly=False):
    print("Getting Stats")

    if on_the_fly:
        print("Welford's algo")
        global_min, global_max = numpy.inf, -numpy.inf
        n = 0
        mean = 0
        m2 = 0

        for file_path in file_paths:
            data = fn_load(file_path)
            global_min = min(global_min, numpy.min(data))
            global_max = max(global_max, numpy.max(data))
            for x in numpy.nditer(data):
                n += 1
                delta = x - mean
                mean += delta / n
                delta2 = x - mean
                m2 += delta * delta2

        variance_n = m2 / n
        stddev = numpy.sqrt(variance_n)
        return global_min, global_max, mean, stddev

    all_data = parallel_style_w_one_arg(func=lambda arg: fn_load(arg), data=file_paths)
    return numpy.mean(all_data), numpy.std(all_data), numpy.amin(all_data), numpy.amax(all_data)


def get_json_file_parameters():
    with open(find_file_cwd("parameters.jsonc"), "r", encoding="utf-8") as file:
        user_parameters = commentjson.load(file)
    return user_parameters


VERBOSE = 0


def parallel_style_w_one_arg(func, data):
    print(f"loading images with {multiprocessing.cpu_count()} cpus")
    with parallel_backend("threading", n_jobs=multiprocessing.cpu_count()):
        return numpy.array(Parallel(prefer="threads", verbose=VERBOSE)(delayed(func)(arg=arg) for arg in data))


def find_directory(name, init_directory, condition, found_path):
    try:
        found_path.append(next(Path(init_directory).rglob(name)))
    except StopIteration:
        pass
    finally:
        with condition:
            condition.notify()


def find_file_cwd(name):
    return next((f for f in Path.cwd().rglob(name)))


def number_clear_cloud(labels_files):
    nb_clear = 0
    nb_cloud = 0
    for _, label in enumerate(labels_files):
        temp = numpy.load(label).item()
        if temp == 0:
            nb_clear += 1
        elif temp == 1:
            nb_cloud += 1
    return nb_clear, nb_cloud
