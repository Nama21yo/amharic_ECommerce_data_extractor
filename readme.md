# Amharic NER for Ethiopian E-Commerce Telegram Channels

## Project Overview

This project develops a Named Entity Recognition (NER) system to extract entities (Product, Price, Location) from Amharic text in Ethiopian e-commerce Telegram channels. It supports EthioMart's micro-lending initiative by analyzing vendor activity through a Vendor Analytics Engine. The project encompasses data collection, preprocessing, labeling, model fine-tuning, comparison, interpretability, and vendor scoring, completed as of June 24, 2025.

### Objectives

1. **Data Ingestion**: Scrape messages from 10 Telegram channels using `Telethon`.
2. **Data Preprocessing**: Normalize and tokenize Amharic text with the `ethiopic` library.
3. **Data Labeling**: Label 30 messages in CoNLL format for NER.
4. **Model Fine-Tuning**: Fine-tune `xlm-roberta-base` for Amharic NER.
5. **Model Comparison**: Compare `xlm-roberta-base`, `distilbert-base-multilingual-cased`, and `bert-base-multilingual-cased` on F1-score, speed, and robustness.
6. **Model Interpretability**: Use SHAP and LIME to explain NER predictions.
7. **Vendor Analytics**: Calculate vendor metrics (posting frequency, views, prices) and a Lending Score.

## Prerequisites

- **Python**: 3.8 or higher
- **Telegram API Credentials**: API ID, API Hash, and Phone Number from [my.telegram.org](https://my.telegram.org)
- **Google Colab**: Recommended for GPU support during model fine-tuning
- **Dependencies**: Listed in `requirements.txt`

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/amharic-ner/amharic-ner-telegram
   cd amharic-ner-telegram
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Telegram API**:

   - Create a `.env` file in the project root:
     ```
     TELEGRAM_API_ID=your_api_id
     TELEGRAM_API_HASH=your_api_hash
     TELEGRAM_PHONE=your_phone_number
     ```

4. **Set Up Directory Structure**:
   ```bash
   mkdir -p data media/photos media/documents models results
   ```

## Project Structure

```
amharic-ner-telegram/
├── data/
│   ├── raw_telegram_data.csv        # Raw Telegram messages with metadata
│   ├── preprocessed_telegram_data.csv  # Preprocessed data with tokenized text
│   ├── labeled_conll.txt            # 30 labeled messages in CoNLL format
│   ├── model_comparison.csv         # Model performance metrics
│   ├── vendor_scorecard.csv         # Vendor analytics and Lending Scores
│   ├── interpretability_report.txt  # SHAP and LIME analysis
│   └── status_report.txt            # Project status summary
├── media/
│   ├── photos/                     # Downloaded photo files
│   └── documents/                  # Downloaded document files
├── models/
│   └── xlm-roberta-ner/            # Fine-tuned NER model
├── requirements.txt                # Project dependencies
├── telegram_scraper.py             # Scrapes Telegram channels
├── preprocess_amharic.py           # Preprocesses Amharic text
├── label_conll.py                  # Labels data in CoNLL format
├── fine_tune_ner.py               # Fine-tunes xlm-roberta-base
├── compare_models.py               # Compares NER models
├── interpretability.py             # Analyzes model predictions with SHAP/LIME
├── vendor_analytics.py             # Computes vendor metrics and scores
├── blog_report.tex                # Blog post summarizing process and results
└── README.md                      # Project documentation
```

## Usage Instructions

### Task 1: Data Ingestion

- **Script**: `telegram_scraper.py`
- **Description**: Scrapes messages from 10 Telegram channels (e.g., @ZemenExpress, @nevacomputer).
- **Run**:
  ```bash
  python telegram_scraper.py
  ```
- **Output**: `data/raw_telegram_data.csv` (messages, metadata like sender, timestamp, views).

### Task 2: Data Preprocessing

- **Script**: `preprocess_amharic.py`
- **Description**: Normalizes and tokenizes Amharic text using the `ethiopic` library.
- **Run**:
  ```bash
  python preprocess_amharic.py
  ```
- **Output**: `data/preprocessed_telegram_data.csv` (tokenized text, metadata).

### Task 3: Data Labeling

- **Script**: `label_conll.py`
- **Description**: Labels 30 messages in CoNLL format for Product, Price, and Location entities.
- **Run**:
  ```bash
  python label_conll.py
  ```
- **Output**: `data/labeled_conll.txt` (CoNLL-formatted dataset).

### Task 4: Model Fine-Tuning

- **Script**: `fine_tune_ner.py`
- **Description**: Fine-tunes `xlm-roberta-base` on `labeled_conll.txt` using Hugging Face's `Trainer` API in Google Colab.
- **Run**:
  ```bash
  python fine_tune_ner.py
  ```
- **Output**: Fine-tuned model saved to `models/xlm-roberta-ner`.

### Task 5: Model Comparison

- **Script**: `compare_models.py`
- **Description**: Fine-tunes and compares `xlm-roberta-base`, `distilbert-base-multilingual-cased`, and `bert-base-multilingual-cased` on F1-score, inference speed, and robustness.
- **Run**:
  ```bash
  python compare_models.py
  ```
- **Output**: `data/model_comparison.csv` (performance metrics).

### Task 6: Model Interpretability

- **Script**: `interpretability.py`
- **Description**: Uses SHAP and LIME to explain NER predictions, analyzing difficult cases.
- **Run**:
  ```bash
  python interpretability.py
  ```
- **Output**: `data/interpretability_report.txt` (SHAP/LIME analysis).

### Task 7: Vendor Analytics Engine

- **Script**: `vendor_analytics.py`
- **Description**: Calculates vendor metrics (posting frequency, average views, top post, average price) and a Lending Score for micro-lending.
- **Run**:
  ```bash
  python vendor_analytics.py
  ```
- **Output**: `data/vendor_scorecard.csv` (vendor metrics and scores).

### Blog Post Report

- **File**: `blog_report.tex`
- **Description**: Summarizes the process, model selection, and NER performance. Compile to PDF using:
  ```bash
  latexmk -pdf blog_report.tex
  ```
- **Output**: `blog_report.pdf`

## Data Summary

- **Channels Scraped**: 10 (e.g., @ZemenExpress, @Shewabrand)
- **Messages Collected**: ~1,000 (100 per channel)
- **Labeled Messages**: 30 in CoNLL format
- **Entities**: Product (B/I-PRODUCT), Price (B/I-PRICE), Location (B/I-LOC)
- **Model Performance**: XLM-RoBERTa (F1: 0.85, best robustness), DistilBERT (fastest), mBERT (balanced)
- **Vendor Metrics**: Posting frequency, average views, top post, average price, Lending Score

## Notes

- **Environment**: Use Google Colab with GPU for fine-tuning (Tasks 4–5). Ensure sufficient disk space for model checkpoints.
- **Data Limitations**: The labeled dataset (30 messages) is small; expand to 100+ for better model performance.
- **Interpretability**: SHAP/LIME highlight ambiguous cases (e.g., modifiers like "ለቦቶ"). Improve with diverse training data.
- **Vendor Analytics**: Assumes `raw_telegram_data.csv` includes views. Adjust script if metadata is incomplete.
- **GitHub**: Code available at [https://github.com/amharic-ner/amharic-ner-telegram](https://github.com/amharic-ner/amharic-ner-telegram).

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License.
