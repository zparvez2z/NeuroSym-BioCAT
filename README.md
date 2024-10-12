# NeuroSym-BioCAT: Leveraging Neuro-Symbolic Methods for Biomedical Scholarly Document Categorization and Question Answering

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

Once the repository and dependencies are set up, you can use **NeuroSym-BioCAT** for document categorization and question answering.

### Running the Categorization and Answer Extraction

```bash
python categorize_and_extract.py --input path_to_input --model minilm --mode abstract
```

Options for the `--mode` argument:
- `abstract`: Retrieve answers from scholarly document abstracts.
- `gold-docs`: Use gold-standard scholarly documents.
- `gold-snippets`: Use gold-standard snippets.

### Fine-Tuning MiniLM (Optional)
```bash
python finetune_minilm.py --data path_to_biomedical_data --epochs 10
```

This will fine-tune the MiniLM model on your domain-specific biomedical dataset.

## Performance and Evaluation

The model has been tested across a variety of biomedical question-answering tasks, demonstrating:
- **Answer Extraction Precision**: Achieves accurate extraction of information with minimal resource overhead.
- **Document Categorization**: High accuracy in categorizing biomedical documents using OVB-LDA with CMA-ES optimization.

Evaluation results and benchmarks can be found in the `results` folder.

## Future Work

Planned improvements for **NeuroSym-BioCAT** include:
- Further optimizing the **MiniLM** model for better performance on complex list-type queries.
- Expanding the topic model with larger, domain-specific datasets.
- Exploring strategies for improving efficiency in answer extraction, especially in real-world applications.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
