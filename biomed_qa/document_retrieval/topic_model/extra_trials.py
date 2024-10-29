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

import optuna
from optuna.storages import RetryFailedTrialCallback


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
            logger.info(f"Created directory: {directory}")


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

    def get_pmids(document):
        return [re.findall(r"/\d+", link)[0][1:] for link in document]

    df["pmids"] = df["documents"].apply(get_pmids)
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
    df = df.copy()

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
    return df, mean_scores_df


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

TRAIN_DATASET_NAME = "Task10BGoldenEnriched"
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

# create directories
create_directories([LOG_DIR, MODEL_DIR])

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


# Define the objective function
def objective(
    trial,
    train_corpus=train_corpus,
    test_corpus=test_corpus,
    test_filtered_df=test_filtered_df,
    test_doc_df=test_doc_df,
    dictionary=dictionary,
    logger=logger,
):
    # Define the hyperparameters to optimize
    num_topics = trial.suggest_int("num_topics", 10, 500)
    chunksize = trial.suggest_categorical(
        "chunksize", [1, 4, 16, 64, 256, 1024, 4096, 16384]
    )
    passes = trial.suggest_int("passes", 1, 50)
    update_every = trial.suggest_int("update_every", 1, 1)
    alpha = trial.suggest_categorical(
        "alpha", ["auto", "symmetric", "asymmetric"]
    )
    eta = trial.suggest_categorical("eta", ["auto", "symmetric"])
    decay = trial.suggest_float("decay", 0.5, 0.9)
    offset = trial.suggest_categorical(
        "offset", [1, 4, 16, 64, 256, 1024, 4096, 16384]
    )
    eval_every = trial.suggest_int("eval_every", 1, 100)
    iterations = trial.suggest_int("iterations", 1, 100)
    gamma_threshold = trial.suggest_float("gamma_threshold", 0.00001, 50.0)
    minimum_probability = trial.suggest_float("minimum_probability", 0.01, 0.01)
    random_state = trial.suggest_int("random_state", 1, 1)
    minimum_phi_value = trial.suggest_float("minimum_phi_value", 0.01, 0.01)

    # Train the LDA model with the suggested hyperparameters
    logger.info(
        "starting training lda model with parameters: num_topics:%s, chunksize:%s, passes:%s, update_every:%s, alpha:%s, eta:%s, decay:%s, offset:%s, eval_every:%s, iterations:%s, gamma_threshold:%s, minimum_probability:%s, random_state:%s, minimum_phi_value:%s",
        num_topics,
        chunksize,
        passes,
        update_every,
        alpha,
        eta,
        decay,
        offset,
        eval_every,
        iterations,
        gamma_threshold,
        minimum_probability,
        random_state,
        minimum_phi_value,
    )
    lda_model = LdaModel(
        corpus=train_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        chunksize=chunksize,
        passes=passes,
        update_every=update_every,
        alpha=alpha,
        eta=eta,
        decay=decay,
        offset=offset,
        eval_every=eval_every,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        minimum_probability=minimum_probability,
        random_state=random_state,
        minimum_phi_value=minimum_phi_value,
    )

    # Create a similarity matrix using the trained LDA model
    logger.info("creating similarity matrix")
    sim_matrix = MatrixSimilarity(
        lda_model[test_corpus], num_features=len(dictionary)
    )

    # get top 10 similar documents for each question
    logger.info("retrieving top similar documents for each question")
    test_filtered_df = test_filtered_df.copy()
    test_filtered_df["top10_docs"] = test_filtered_df[
        "body_preprocessed"
    ].apply(retrieve_documents, args=(lda_model, sim_matrix))

    test_filtered_df["top10_pmids"] = test_filtered_df["top10_docs"].apply(
        get_pmids_from_doc_indexes, args=(test_doc_df,)
    )

    # calculate metrics
    logger.info("calculating metrics")
    eval_df, eval_df_summary = calculate_metrics(
        test_filtered_df, "pmids_found", "top10_pmids"
    )
    # Return the mean mean_f1 score
    return eval_df_summary["mean_f1"].iloc[0]


server_type = "local"
# server_type = "azure"

if server_type == "local":
    # define storage using local postgresql database
    storage = optuna.storages.RDBStorage(
        url="postgresql://user:password@localhost:5432/app",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )
if server_type == "azure":
    # define storage using azure postgresql database
    storage = optuna.storages.RDBStorage(
        url="postgresql://optuna:pwd@server.postgres.database.azure.com:5432/optunadb",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

STUDY_NAME = f"optimize_lda_{TRAIN_DATASET_NAME}_on_{TEST_DATASET_NAME}"
print(f"study name:{STUDY_NAME}")
logger.info("study name:%s", STUDY_NAME)

# Set up the Optuna study or load an existing one
study = optuna.create_study(
    study_name=STUDY_NAME,
    direction="maximize",
    storage=storage,
    load_if_exists=True,
)

# #  fix the sampling parameters
# study.enqueue_trial(
#     {
#         "num_topics": 150,
#         "chunksize": 1024,
#         "passes": 10,
#         "update_every": 1,
#         "alpha": "auto",
#         "eta": "auto",
#         "decay": 0.5,
#         "offset": 1,
#         "eval_every": 10,
#         "iterations": 50,
#         "gamma_threshold": 50.0,
#         "minimum_probability": 0.01,
#         "random_state": 1,
#         "minimum_phi_value": 0.01,
#     }
# )

# Optimize the hyperparameters using Optuna
study.optimize(
    objective,
    n_trials=10,
    catch=(ValueError),
    gc_after_trial=True,
    show_progress_bar=True,
)

# Retrieve the best hyperparameters and corresponding score
best_params = study.best_params
best_score = study.best_value

# Print the best hyperparameters and score
print("Best Hyperparameters: ", best_params)
logger.info("Best Hyperparameters: %s", best_params)
print("Best Score: ", best_score)
logger.info("Best Score: %s", best_score)
