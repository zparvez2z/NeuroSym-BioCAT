"""this module contains common functions used in the project"""
import os
import json
import logging
import glob
import pandas as pd
import re
from itertools import chain

logger = logging.getLogger(__name__)


def create_directories(directories):
    """create directories if they do not exist"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def init_directories(
    FILE_DIR, DATASET, YEAR, TRAIN_DATASET_NAME, TEST_DATASET_NAME
):
    """init directories"""
    try:
        # load dir_dict from json file in home directory
        home_dir = os.path.expanduser("~")
        with open(f"{home_dir}/.biomedqa_dir.json", encoding="utf-8") as fp:
            dir_dict = json.load(fp)
    except Exception as exc:
        print("Error: unable to load directory dictionary. Please run setup.py")
        raise exc
    # set directories
    BASE_DIR = dir_dict["base_dir"]
    DATA_DIR = dir_dict["data_dir"]
    MODEL_DIR = dir_dict["model_dir"]
    LOG_DIR = dir_dict["log_dir"]
    RESULTS_DIR = dir_dict["results_dir"]

    TRAIN_DATASET_DIR = f"{DATA_DIR}/raw/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}"
    TRAIN_DOC_DIR = (
        f"{DATA_DIR}/processed/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}_documents/"
    )
    print(f"train dataset name:{TRAIN_DATASET_NAME}")
    print(f"train dataset dir:{TRAIN_DATASET_DIR}")
    print(f"train doc dir:{TRAIN_DOC_DIR}")

    TEST_DATASET_DIR = f"{DATA_DIR}/raw/{DATASET}/{YEAR}/{TEST_DATASET_NAME}"
    TEST_DOC_DIR = (
        f"{DATA_DIR}/processed/{DATASET}/{YEAR}/{TEST_DATASET_NAME}_documents/"
    )
    print(f"test dataset name:{TEST_DATASET_NAME}")
    print(f"test dataset dir:{TEST_DATASET_DIR}")
    print(f"test doc dir:{TEST_DOC_DIR}")

    # set log dir directory according to current file directory
    LOG_DIR = f"{LOG_DIR}/{FILE_DIR}"
    print(f"log dir:{LOG_DIR}")

    # set model directory according to current file directory
    MODEL_DIR = f"{MODEL_DIR}/{FILE_DIR}/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}/"
    print(f"model dir:{MODEL_DIR}")

    # create directories
    create_directories([LOG_DIR, MODEL_DIR])

    return (
        BASE_DIR,
        DATA_DIR,
        MODEL_DIR,
        LOG_DIR,
        RESULTS_DIR,
        TRAIN_DATASET_DIR,
        TRAIN_DOC_DIR,
        TEST_DATASET_DIR,
        TEST_DOC_DIR,
    )


def init_logger(LOG_DIR, file_name):
    """init logger"""
    # set log file name
    log_file = os.path.join(
        LOG_DIR, os.path.basename(file_name).split(".")[0] + ".log"
    )
    print(f"log_file: {log_file}")

    # initialize logger
    logging.basicConfig(
        filename=log_file,
        format="%(process)d\t%(asctime)s\t%(levelname)s\t%(message)s",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logger initialized")

    return logger

def load_dataset(dataset_name, dataset_dir):
    dfs = []
    for json_file in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(json_file) as fp:
            json_data = "".join(fp)
        data = json.loads(json_data)
        data = data["questions"]
        dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_dataset(dataset_name, dataset_dir, doc_df):
    dfs = []
    for json_file in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(json_file) as fp:
            json_data = "".join(fp)
        data = json.loads(json_data)
        data = data["questions"]
        dfs.append(pd.DataFrame(data))
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)

    def extract_pmid(links):
        return [link.split("/")[-1] for link in links]

    df["pmids"] = df["documents"].apply(extract_pmid)
    pmids = list(chain.from_iterable(df["pmids"].to_list()))
    print(
        f"total number of unique docs provided in {dataset_name}: {len(set(pmids))}"
    )
    logger.info(
        "total number of unique docs provided in %s: %d",
        dataset_name,
        len(set(pmids)),
    )

    corpus_df = doc_df[doc_df["pmid"].isin(pmids)]
    print(f"num of docs found in corpus:{corpus_df.shape[0]}")
    logger.info("num of docs found in corpus: %d", corpus_df.shape[0])

    def filter_pmid(pmids):
        filtered_pmids = [
            pmid for pmid in pmids if pmid in doc_df["pmid"].to_list()
        ]
        return filtered_pmids

    df["pmids_found"] = df["pmids"].apply(filter_pmid)

    filtered_df = df[df["pmids_found"].apply(len) > 0]
    # average number of docs per query
    total_num_docs = sum(filtered_df["pmids_found"].apply(len))
    total_num_queries = filtered_df["body"].shape[0]
    avg_num_docs_per_query = total_num_docs / total_num_queries
    print(f"avg num of docs per query: {avg_num_docs_per_query}")
    logger.info("avg num of docs per query: %d", avg_num_docs_per_query)
    return filtered_df
