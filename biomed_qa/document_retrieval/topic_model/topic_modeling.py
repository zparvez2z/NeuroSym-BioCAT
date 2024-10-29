# import libraries
import datetime
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
from gensim.models.callbacks import Metric

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import FrozenTrial


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


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


def calculate_metrics(df, pred_col_name):
    golden = []
    predicted = []

    for i in range(len(df)):
        golden.append({
            "id": df.iloc[i]["id"],
            "body": df.iloc[i]["body"],
            "type": df.iloc[i]["type"],
            "documents": df.iloc[i]["documents"],
            "snippets": [],
            "ideal_answer": [],
            "exact_answer": [],
        })

        predicted.append({
            "id": df.iloc[i]["id"],
            "body": df.iloc[i]["body"],
            "type": df.iloc[i]["type"],
            "documents": df.iloc[i][pred_col_name],
            "snippets": [],
            "ideal_answer": [],
            "exact_answer": [],
        })

    with open("golden.json", "w") as fp:
        json.dump({"questions": golden}, fp, indent=4)

    with open("predicted.json", "w") as fp:
        json.dump({"questions": predicted}, fp, indent=4)

    evaluation_command = 'java -Xmx10G -cp "/workspaces/biomed_qa-zparvez2z/Evaluation-Measures/flat/BioASQEvaluation/dist/*" evaluation.EvaluatorTask1b -phaseA -e 5 golden.json predicted.json -verbose'
    evaluation_output = os.popen(evaluation_command).read()        
    evaluation_output = evaluation_output.strip().split("\n")
    score_dict = {}
    for line in evaluation_output[1:]:
            metric, score = line.split(":")
            score_dict[metric.strip()] = float(score.strip())

    return pd.DataFrame(score_dict, index=[0]) 


def evaluate(
    lda_model, test_corpus, test_df, test_doc_df, dictionary, metric, logger
):
    # Create a similarity matrix using the trained LDA model
    logger.info("creating similarity matrix")
    sim_matrix = MatrixSimilarity(
        lda_model[test_corpus], num_features=len(dictionary)
    )

    # get top 10 similar documents for each question
    logger.info("retrieving top similar documents for each question")
    test_df = test_df.copy()
    test_df["top10_sims"] = test_df["body_preprocessed"].apply(
        retrieve_documents, args=(lda_model, sim_matrix)
    )

    test_df["top10_pmids"] = test_df["top10_sims"].apply(
        get_pmids_from_doc_indexes, args=(test_doc_df,)
    )

    test_df["top10_docs"] = test_df["top10_pmids"].apply(
        lambda docs: ["http://www.ncbi.nlm.nih.gov/pubmed/" + str(pmid) for pmid in docs]
    )
    # calculate metrics
    logger.info("calculating metrics")
    eval_df_summary = calculate_metrics(test_df, "top10_docs")
    # Return the metric score
    score = eval_df_summary[metric].iloc[0]       
    logger.info("%s: %s", metric, score)
    return score


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
# train_doc_df = pickle.load(
#     open(f"{TRAIN_DOC_DIR}{TRAIN_DATASET_NAME}_documents_df.pkl", "rb")
# )
train_doc_df = pd.read_pickle(
    f"{TRAIN_DOC_DIR}{TRAIN_DATASET_NAME}_documents_df.pkl"
)

# test_doc_df = pickle.load(
#     open(f"{TEST_DOC_DIR}{TEST_DATASET_NAME}_documents_df.pkl", "rb")
# )
test_doc_df = pd.read_pickle(
    f"{TEST_DOC_DIR}{TEST_DATASET_NAME}_documents_df.pkl"
)

# prepare the initial datset
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


# custom gensim callback to store intermediate scores and add the best result to the optuna study
class IntermediateScoreCallback(Metric):
    """Callback to log information about training"""

    def __init__(
        self,
        test_corpus,
        test_df,
        test_doc_df,
        dictionary,
        trial,
        optimization_metric,
        logger=None,
        title=None,
    ):
        self.logger = logger
        self.title = title
        self.pass_count = 0
        self.test_corpus = test_corpus
        self.test_df = test_df
        self.test_doc_df = test_doc_df
        self.dictionary = dictionary
        self.trial = trial
        self.datetime_start = datetime.datetime.now()
        self.optimization_metric = optimization_metric
        self.score = None

    def get_value(self, **kwargs):
        super(IntermediateScoreCallback, self).set_parameters(**kwargs)
        self.pass_count += 1
        # train_and_evaluate(self,test_dataset_name,test_filtered_df,test_doc_df, **kwargs)
        self.score = evaluate(
            lda_model=self.model,
            test_corpus=self.test_corpus,
            test_df=self.test_df,
            test_doc_df=self.test_doc_df,
            dictionary=self.dictionary,
            metric=self.optimization_metric,
            logger=self.logger,
        )
        # add intermediate score to optuna study with start and end time
        intermediate_trial = optuna.trial.create_trial(
            params=self.trial.params,
            distributions=self.trial.distributions,
            value=self.score,
            user_attrs=None,
            system_attrs=None,
            state=optuna.trial.TrialState.COMPLETE,
        )
        intermediate_trial.params["passes"] = self.pass_count
        intermediate_trial.datetime_start = self.datetime_start
        intermediate_trial.datetime_complete = datetime.datetime.now()
        study.add_trial(intermediate_trial)


# Define the objective function
def objective(
    trial,
    train_corpus=train_corpus,
    test_corpus=test_corpus,
    test_filtered_df=test_filtered_df,
    test_doc_df=test_doc_df,
    dictionary=dictionary,
    optimization_metric="MAP documents",
    logger=logger,
):
    # Define the hyperparameters to optimize
    num_topics = trial.suggest_int("num_topics", 50, 2000)
    chunksize = trial.suggest_int("chunksize", 1, len(train_corpus))
    passes = trial.suggest_int("passes", 1, 50)
    update_every = 1
    alpha = "symmetric"
    eta = "symmetric"
    decay = 0.5
    offset = 1
    eval_every = 10
    iterations = trial.suggest_int("iterations", 1, 100)
    gamma_threshold = 0.001
    minimum_probability = 0.01
    random_state = 1
    minimum_phi_value = 0.01

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

    intermediate_score_callback = IntermediateScoreCallback(
        test_corpus=test_corpus,
        test_df=test_filtered_df,
        test_doc_df=test_doc_df,
        dictionary=dictionary,
        trial=trial,
        optimization_metric=optimization_metric,
        logger=logger,
        title="LDA",
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
        callbacks=[intermediate_score_callback],
    )

    return intermediate_score_callback.score


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
# sampler = optuna.samplers.CmaEsSampler(
#     seed=1, n_startup_trials=3, restart_strategy="ipop"
# )

sampler = optuna.samplers.TPESampler(
    seed=1, n_startup_trials=2, multivariate=True
)

STUDY_NAME = (
    f"optimize_lda_{TRAIN_DATASET_NAME}_on_{TEST_DATASET_NAME}"
    + f"_{sampler.__class__.__name__}"
    # + f"_{sampler._restart_strategy}"
    + f"_v2"
)
print(f"study name:{STUDY_NAME}")
logger.info("study name:%s", STUDY_NAME)

# Set up the Optuna study oror load an existing one
study = optuna.create_study(
    study_name=STUDY_NAME,
    sampler=sampler,
    direction="maximize",
    storage=storage,
    load_if_exists=True,
)

# Run the optimization
study.optimize(
    objective,
    n_trials=500,
    catch=(ValueError),
    gc_after_trial=True,
    show_progress_bar=True,
)

#Retrieve the best hyperparameters and corresponding score
best_params = study.best_params
best_score = study.best_value

# Print the best hyperparameters and score
print("Best Hyperparameters: ", best_params)
logger.info("Best Hyperparameters: %s", best_params)
print("Best Score: ", best_score)
logger.info("Best Score: %s", best_score)