# import libraries
import logging
import json
import pickle
import glob
import os
import re
from itertools import chain
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from scipy.stats import gmean

# import gensim libraries
from gensim import corpora
from gensim.models import LdaModel

# from gensim.models.ldamulticore import LdaMulticore
from gensim.similarities import MatrixSimilarity
from gensim.parsing.preprocessing import preprocess_documents


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def extract_pmid(links):
    return [link.split("/")[-1] for link in links]


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


# retrieve the top N similar documents for a given document or query
def retrieve_documents(query, lda_model, sim_matrix, topn=10):
    vec_bow = dictionary.doc2bow(query)
    vec_lda = lda_model[vec_bow]
    sims = sim_matrix[vec_lda]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims[:topn]


# get pmids from doc indexes
def get_pmids_from_doc_indexes(doc_indexes, doc_df):
    return [doc_df["pmid"].iloc[doc_idx[0]] for doc_idx in doc_indexes]


def calculate_metrics(df, true_col, pred_col):
    # Calculate precision, recall, f1, and average precision for each row
    df["precision"] = 0
    df["recall"] = 0
    df["f1"] = 0
    df["avg_precision"] = 0

    for i in range(len(df)):
        # Fit MultiLabelBinarizer on each row separately
        mlb = MultiLabelBinarizer()
        mlb.fit(
            [df[true_col].iloc[i] + df[pred_col].iloc[i]]
        )  # Combining true and predicted labels

        # Transform true and predicted columns separately
        X_true = mlb.transform([df[true_col].iloc[i]])
        X_pred = mlb.transform([df[pred_col].iloc[i]])

        # Calculate precision, recall, f1, and average precision for the current row
        df.at[i, "precision"] = precision_score(
            X_true[0], X_pred[0], zero_division=0
        )
        df.at[i, "recall"] = recall_score(X_true[0], X_pred[0], zero_division=0)
        df.at[i, "f1"] = f1_score(X_true[0], X_pred[0], zero_division=0)
        df.at[i, "avg_precision"] = average_precision_score(
            X_true[0], X_pred[0]
        )

    # Calculate mean precision, mean recall, and mean f1
    mean_precision = df["precision"].mean()
    mean_recall = df["recall"].mean()
    mean_f1 = df["f1"].mean()

    # Calculate MAP and GMAP
    map_score = df["avg_precision"].mean()
    gmap_score = gmean(df["avg_precision"])

    # Create a new dataframe to store the mean scores
    mean_scores_df = pd.DataFrame(
        {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "MAP": map_score,
            "GMAP": gmap_score,
        },
        index=[0],
    )

    # Return both dataframes
    return mean_scores_df


def get_max(logs, metric):
    df = pd.concat(logs)
    print(f"max {metric}:")
    df = df.sort_values(by=[metric], ascending=False)
    # return df[df[metric] == df[metric].max()]
    return df.head()


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

DATASET = "bioasq"
YEAR = "2022"

TRAIN_DATASET_NAME = "BioASQ-training10b"
TRAIN_DATASET_DIR = f"{DATA_DIR}/raw/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}"
TRAIN_DOC_DIR = (
    f"{DATA_DIR}/processed/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}_documents/"
)
print(f"train dataset name:{TRAIN_DATASET_NAME}")
print(f"train dataset dir:{TRAIN_DATASET_DIR}")
print(f"train doc dir:{TRAIN_DOC_DIR}")

TEST_DATASET_NAME = "Task10BGoldenEnriched"
TEST_DATASET_DIR = f"{DATA_DIR}/raw/{DATASET}/{YEAR}/{TEST_DATASET_NAME}"
TEST_DOC_DIR = (
    f"{DATA_DIR}/processed/{DATASET}/{YEAR}/{TEST_DATASET_NAME}_documents/"
)
print(f"test dataset name:{TEST_DATASET_NAME}")
print(f"test dataset dir:{TEST_DATASET_DIR}")
print(f"test doc dir:{TEST_DOC_DIR}")

# get file directory
FILE_DIR = os.path.dirname(os.path.relpath(__file__))

# set log dir directory according to current file directory
LOG_DIR = f"{LOG_DIR}/{FILE_DIR}"
print(f"log dir:{LOG_DIR}")

# set model directory according to current file directory
MODEL_DIR = f"{MODEL_DIR}/{FILE_DIR}/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}/"
print(f"model dir:{MODEL_DIR}")

# set results directory according to current file directory
RESULTS_DIR = f"{RESULTS_DIR}/{FILE_DIR}/{DATASET}/{YEAR}/{TRAIN_DATASET_NAME}/"
print(f"results dir:{RESULTS_DIR}")

# create directories
create_directories([LOG_DIR, MODEL_DIR, RESULTS_DIR])

# set log file name
log_file = os.path.join(
    LOG_DIR, os.path.basename(__file__).split(".")[0] + ".log"
)
print(f"LOG_FILE: {log_file}")

# initialize logger

logging.basicConfig(
    filename=log_file,
    format="%(process)d\t%(asctime)s\t%(levelname)s\t%(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized")

# load documents
logger.info("loading documents")
train_doc_df = pickle.load(
    open(f"{TRAIN_DOC_DIR}{TRAIN_DATASET_NAME}_documents_df.pkl", "rb")
)

test_doc_df = pickle.load(
    open(f"{TEST_DOC_DIR}{TEST_DATASET_NAME}_documents_df.pkl", "rb")
)

train_filtered_df = prepare_dataset(
    TRAIN_DATASET_NAME, TRAIN_DATASET_DIR, train_doc_df
)
test_filtered_df = prepare_dataset(
    TEST_DATASET_NAME, TEST_DATASET_DIR, test_doc_df
)

# preprocess documents using gensim's preprocess_documents function
logger.info("preprocessing train documents")
train_doc_df["abstractText_preprocessed"] = preprocess_documents(
    train_doc_df["abstractText"]
)
logger.info("preprocessing test documents")
test_doc_df["abstractText_preprocessed"] = preprocess_documents(
    test_doc_df["abstractText"]
)

# Create a dictionary from the preprocessed documents of the training set
logger.info("creating dictionary")
dictionary = corpora.Dictionary(train_doc_df["abstractText_preprocessed"])

# create bag of words corpus of the training set
logger.info("creating bag of words for train documents")
train_corpus = [
    dictionary.doc2bow(text)
    for text in train_doc_df["abstractText_preprocessed"]
]
# Create bag of words corpus of the test set
logger.info("creating bag of words for test documents")
test_corpus = [
    dictionary.doc2bow(text)
    for text in test_doc_df["abstractText_preprocessed"]
]

# preprocess questions
logger.info("preprocessing test questions")
test_filtered_df["body_preprocessed"] = preprocess_documents(
    test_filtered_df["body"].to_list()
)

# train LDA model
lda_model = LdaModel(
    corpus=train_corpus,
    id2word=dictionary,
    num_topics=1390,
    chunksize=2510,
    passes=5,
    update_every=1,
    alpha="symmetric",
    eta="symmetric",
    decay=0.5,
    offset=1,
    eval_every=10,
    iterations=256,
    gamma_threshold=0.001,
    minimum_probability=0.01,
    random_state=1,
    minimum_phi_value=0.01,
)

# Create a similarity matrix using the trained LDA model
logger.info("creating similarity matrix")
sim_matrix = MatrixSimilarity(
    lda_model[test_corpus], num_features=len(dictionary)
)

# get top 10 similar documents for each question
logger.info("retrieving top similar documents for each question")
test_df = test_filtered_df.copy()
test_df["top10_docs"] = test_df["body_preprocessed"].apply(
    retrieve_documents, args=(lda_model, sim_matrix)
)

test_df["top10_pmids"] = test_df["top10_docs"].apply(
    get_pmids_from_doc_indexes, args=(test_doc_df,)
)

# calculate metrics
logger.info("calculating metrics")
eval_df_summary = calculate_metrics(test_df, "pmids_found", "top10_pmids")

# calculate metrics for each batch
logger.info("calculating metrics for each batch")
score_dfs = []
for batch in glob.glob(f"{TEST_DATASET_DIR}/*.json"):
    batch_name = os.path.basename(batch).split(".")[0]
    with open(batch) as f:
        batch_json = json.load(f)
    batch_df = pd.DataFrame(batch_json["questions"])
    batch_df["body_preprocessed"] = preprocess_documents(
        batch_df["body"].to_list()
    )
    batch_df["pmids"] = batch_df["documents"].apply(extract_pmid)
    batch_df["top10_docs"] = batch_df["body_preprocessed"].apply(
        retrieve_documents, args=(lda_model, sim_matrix)
    )
    batch_df["top10_pmids"] = batch_df["top10_docs"].apply(
        get_pmids_from_doc_indexes, args=(test_doc_df,)
    )
    batch_eval_df = calculate_metrics(batch_df, "pmids", "top10_pmids")
    batch_eval_df["batch"] = batch_name
    score_dfs.append(batch_eval_df)

score_df = pd.concat(score_dfs)
score_df = score_df.sort_values(by=["batch"])

# save results
logger.info("saving results")
eval_df_summary.to_csv(
    f"{RESULTS_DIR}{TRAIN_DATASET_NAME}_results.csv", index=False
)
score_df.to_csv(
    f"{RESULTS_DIR}{TRAIN_DATASET_NAME}_results_by_batch.csv", index=False
)
