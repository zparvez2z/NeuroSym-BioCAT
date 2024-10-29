# NeuroSym-BioCAT

## Introduction
This repository contains the code and documentation of our extensive research and experimentaton on developing a neuro-symbolic scholarly document categorization based retrieval and question answering expected to deliver high-accuracy responses to a variety of biomedical queries. The system operates in two main parts: document retrieval and answer extraction, with several methods employed for each part. 
Biomedical research generates an immense amount of scholarly documents, making it challenging for researchers to efficiently extract relevant information. **NeuroSym-BioCAT** aims to address these challenges by:
- Optimizing topic modeling using **OVB-LDA** with advanced optimization techniques for document categorization.
- Leveraging the **distilled MiniLM model** fine-tuned on domain-specific data for high-precision biomedical answer extraction.

This repository demonstrates how integrating neuro-symbolic methods can improve efficiency and accuracy in scholarly document abstract retrieval and question answering tasks, outperforming existing methods such as **RYGH** and **bio-answerfinder**.

## Installation

To set up this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NeuroSym-BioCAT.git
   cd NeuroSym-BioCAT
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Run init.py to set up all the dependencies like install required python packeges and directories:
   ```bash
   python init.py
   ```

## Usage

### 1. **Data Preprocessing**
Start by enhancing the dataset by fetching PubMed abstracts using the PubMed IDs from the BioASQ10 dataset. run the [bioasq_doc_prep.ipynb](notebooks/bioasq_doc_prep.ipynb)

### 2. **Document Categorization**
To categorize scholarly documents, run the [topic_modeling.ipynb](notebooks/topic_modeling.ipynb)
or 
```bash
python biomed_qa/document_retrieval/topic_model/topic_modeling.py
```

### 3. **Answer Extraction**
For fine-tuning the MiniLM model and answer extraction run [answer_extraction.ipynb](notebooks/answer_extraction.ipynb). or
```bash
python biomed_qa/answer_extraction/transformer_based/answer_extraction.py
```

### 4. **Evaluation**
Evaluate the document categorization performance see [evaluate_by_batch.ipynb](notebooks/evaluate_by_batch.ipynb)

Evaluate the fines-tuned model performance using precision, recall, F1-score for list-type questions, and MRR, strict, and lenient accuracy for factoid questions see [answer_extraction_evaluation.ipynb](notebooks/answer_extraction_evaluation.ipynb)

## Future Work

Planned improvements for **NeuroSym-BioCAT** include:
- Task 1
- Task 2
- Task 3

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
