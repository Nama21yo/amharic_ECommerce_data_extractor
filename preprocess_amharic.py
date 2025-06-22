import pandas as pd
from ethiopic import normalize, tokenize
import re

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    # Normalize Amharic text (e.g., unify labialized forms)
    normalized_text = normalize(text)
    # Tokenize text
    tokens = tokenize(normalized_text)
    # Remove special characters and extra spaces
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if token.strip()]
    return tokens

def preprocess_data():
    # Load raw data
    df = pd.read_csv('data/raw_telegram_data.csv', encoding='utf-8')
    
    # Preprocess text
    df['tokens'] = df['message'].apply(preprocess_text)
    
    # Clean metadata
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['channel'] = df['channel'].str.strip()
    
    # Save preprocessed data
    df.to_csv('data/preprocessed_telegram_data.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    preprocess_data()
