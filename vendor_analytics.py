import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta

# Load fine-tuned NER model
model_path = "models/xlm-roberta-ner"
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")

# Load raw data
data = pd.read_csv('data/raw_telegram_data.csv', encoding='utf-8')

# Extract entities
def extract_entities(text):
    if not isinstance(text, str):
        return {'products': [], 'prices': [], 'locations': []}
    entities = ner_pipeline(text)
    products, prices, locations = [], [], []
    for entity in entities:
        if entity['entity_group'].startswith('PRODUCT'):
            products.append(entity['word'])
        elif entity['entity_group'].startswith('PRICE'):
            prices.append(entity['word'])
        elif entity['entity_group'].startswith('LOC'):
            locations.append(entity['word'])
    return {'products': products, 'prices': prices, 'locations': locations}
