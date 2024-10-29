""" This file contains the code for the transformer based answer extraction method utilizing """
import logging
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from biomed_qa.utils.common import (
    init_directories,
    init_logger,
    load_dataset,
)

# init directories
# get file directory
FILE_DIR = os.path.dirname(os.path.relpath(__file__))
DATASET = "bioasq"
YEAR = "2022"
TRAIN_DATASET_NAME = "BioASQ-training10b"
TEST_DATASET_NAME = "Task10BGoldenEnriched"

(
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    RESULTS_DIR,
    TRAIN_DATASET_DIR,
    TRAIN_DOC_DIR,
    TEST_DATASET_DIR,
    TEST_DOC_DIR,
) = init_directories(
    FILE_DIR, DATASET, YEAR, TRAIN_DATASET_NAME, TEST_DATASET_NAME
)

logger = init_logger(LOG_DIR, __file__)

# load documents
logger.info("loading documents")
train_doc_df = pickle.load(
    open(f"{TRAIN_DOC_DIR}{TRAIN_DATASET_NAME}_documents_df.pkl", "rb")
)

test_doc_df = pickle.load(
    open(f"{TEST_DOC_DIR}{TEST_DATASET_NAME}_documents_df.pkl", "rb")
)

train_df = load_dataset(TRAIN_DATASET_NAME, TRAIN_DATASET_DIR)
test_df = load_dataset(TEST_DATASET_NAME, TEST_DATASET_DIR)

def create_squad_dataset(doc_df, df):
    squad_data = {"data": []}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        query = row["query"]
        pmids = row["PMIDs"]
        exact_answers = row["exact_answers"]

        squad_paragraphs = []

        for pmid in pmids:
            abstract_text = doc_df.loc[
                doc_df["PMID"] == pmid, "abstractText"
            ].iloc[0]

            paragraph = {"context": abstract_text, "qas": []}

            for answer in exact_answers:
                qas = {
                    "question": query,
                    "id": f"{pmid}_{query}",
                    "answers": [],
                }

                if isinstance(answer, list):
                    for ans in answer:
                        ans_lower = ans.lower()
                        start_idx = abstract_text.lower().find(ans_lower)
                        if start_idx != -1:
                            qas["answers"].append(
                                {"text": ans, "answer_start": start_idx}
                            )
                else:
                    ans_lower = answer.lower()
                    start_idx = abstract_text.lower().find(ans_lower)
                    if start_idx != -1:
                        qas["answers"].append(
                            {"text": answer, "answer_start": start_idx}
                        )

                paragraph["qas"].append(qas)

            squad_paragraphs.append(paragraph)

        squad_data["data"].append({"title": "", "paragraphs": squad_paragraphs})

    return squad_data

# function to filter factoid questions from the dataset
def filter_factoid_questions(df):
    return df[df["type"] == "factoid"]


# function to filter list questions from the dataset
def filter_list_questions(df):
    return df[df["type"] == "list"]


# filter factoid questions from train_filtered_df
train_filtered_factoid_df = filter_factoid_questions(train_df)

# filter list questions from train_filtered_df
train_filtered_list_df = filter_list_questions(train_df)

# filter factoid questions from test_filtered_df
test_filtered_factoid_df = filter_factoid_questions(test_df)

# filter list questions from test_filtered_df
test_filtered_list_df = filter_list_questions(test_df)
