# NeuroSym-BioCAT

## Introduction
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

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained models (optional):
   - You can download and fine-tune the MiniLM model from the Hugging Face model hub.

## Usage

### 1. **Data Preprocessing**
Start by enhancing the dataset by fetching PubMed abstracts using the PubMed IDs from the BioASQ10 dataset. Run the preprocessing step to clean the data by removing stop words, punctuation, special characters, and applying stemming for standardization.

```bash
python preprocess_data.py --input bioasq10_dataset --output preprocessed_data
```

### 2. **Document Categorization**
To categorize scholarly documents, run the optimized OVB-LDA model with BI-POP CMA-ES for parameter tuning.

```bash
python categorize_documents.py --input preprocessed_data --output categorized_docs
```

### 3. **Answer Extraction**
Extract answers from categorized documents using the fine-tuned MiniLM model. The system can handle factoid and list-type questions.

```bash
python extract_answers.py --input categorized_docs --questions bioasq10_questions --output answers
```

### 4. **Evaluation**
Evaluate the model performance using precision, recall, F1-score for list-type questions, and MRR, strict, and lenient accuracy for factoid questions.

```bash
python evaluate_model.py --answers extracted_answers --golden bioasq10_golden --metrics output_metrics
``` 


## Future Work

Planned improvements for **NeuroSym-BioCAT** include:
- Task 1
- Task 2
- Task 3

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
