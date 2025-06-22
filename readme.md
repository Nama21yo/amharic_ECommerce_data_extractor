Amharic NER for E-Commerce Telegram Channels
Overview
This project focuses on building a Named Entity Recognition (NER) system to extract entities (Product, Price, Location) from Amharic text in Ethiopian e-commerce Telegram channels. The project includes scripts for data ingestion, preprocessing, and labeling in CoNLL format, with plans for fine-tuning and comparing NER models. As of June 22, 2025, the project has implemented data collection from 10 Telegram channels, preprocessing for Amharic text, and labeling of 30 messages.
Project Objectives

Data Ingestion: Collect messages (text, images, documents) from Ethiopian e-commerce Telegram channels.
Data Preprocessing: Normalize and tokenize Amharic text, preparing it for NER tasks.
Data Labeling: Label a subset of messages in CoNLL format for Product, Price, and Location entities.
Future Work: Fine-tune NER models (e.g., BERT) and compare performance (e.g., BERT vs. BiLSTM-CRF).

Prerequisites

Python 3.8+
Telegram API credentials (API ID, API Hash, Phone Number)
Required Python libraries (see requirements.txt)

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd amharic-ner-telegram

Install Dependencies:
pip install -r requirements.txt

Set Up Telegram API Credentials:

Create a .env file in the project root.
Add the following:TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=your_phone_number

Obtain credentials from my.telegram.org.

Create Directory Structure:
mkdir -p data media/photos media/documents

Project Structure
amharic-ner-telegram/
├── data/
│ ├── raw_telegram_data.csv # Raw scraped Telegram messages
│ ├── preprocessed_telegram_data.csv # Preprocessed data with tokens
│ ├── labeled_conll.txt # Labeled dataset in CoNLL format
│ └── status_report.txt # Project status report
├── media/
│ ├── photos/ # Downloaded photo files
│ └── documents/ # Downloaded document files
├── requirements.txt # Project dependencies
├── telegram_scraper.py # Script for scraping Telegram channels
├── preprocess_amharic.py # Script for preprocessing Amharic text
├── label_conll.py # Script for CoNLL labeling
└── README.md # Project documentation

Usage

Data Ingestion:

Run the Telegram scraper to collect messages from 10 channels:python telegram_scraper.py

Output: data/raw_telegram_data.csv with messages and metadata.

Data Preprocessing:

Preprocess Amharic text for NER:python preprocess_amharic.py

Output: data/preprocessed_telegram_data.csv with tokenized text.

Data Labeling:

Label 30 messages in CoNLL format:python label_conll.py

Output: data/labeled_conll.txt with labeled entities.

View Status Report:

Review the project status in data/status_report.txt.

Data Summary

Channels: 10 (e.g., @ZemenExpress, @nevacomputer, @Shewabrand)
Messages Collected: ~1,000 (100 per channel, subject to filtering)
Labeled Messages: 30 in CoNLL format
Entities: Product (B-Product, I-Product), Price (B-Price, I-Price), Location (B-Location, I-Location)
Storage: Raw and preprocessed data in CSV; labeled data in labeled_conll.txt

Notes

The Telegram scraper requires valid API credentials and a stable internet connection.
The labeling script uses a dummy function for demonstration; replace with manual annotation or tools like doccano for production.
The labeled dataset is small (30 messages); expand to 100+ messages for robust model training.
Future steps include fine-tuning a BERT-based model and comparing NER performance.

Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, features, or improvements.
License
This project is licensed under the MIT License.
