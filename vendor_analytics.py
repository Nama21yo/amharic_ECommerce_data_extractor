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

# Calculate metrics
vendor_metrics = []
channels = data['channel'].unique()
for channel in channels:
    vendor_data = data[data['channel'] == channel]
    vendor_data['timestamp'] = pd.to_datetime(vendor_data['timestamp'])
    
    # Posting frequency (posts per week)
    time_span = (vendor_data['timestamp'].max() - vendor_data['timestamp'].min()).days / 7
    posts_per_week = len(vendor_data) / max(time_span, 1)
    
    # Average views per post
    avg_views = vendor_data['views'].mean() if 'views' in vendor_data.columns else 0
    
    # Top performing post
    top_post = vendor_data.loc[vendor_data['views'].idxmax()] if 'views' in vendor_data.columns else vendor_data.iloc[0]
    top_post_entities = extract_entities(top_post['message'])
    
    # Average price point
    prices = []
    for _, row in vendor_data.iterrows():
        entities = extract_entities(row['message'])
        for price in entities['prices']:
            try:
                price_num = float(''.join(filter(str.isdigit, price)))
                prices.append(price_num)
            except:
                pass
    avg_price = np.mean(prices) if prices else 0
    
    # Lending Score
    lending_score = (avg_views * 0.5) + (posts_per_week * 0.5)
    
    vendor_metrics.append({
        'Vendor': channel,
        'Avg_Views_Per_Post': avg_views,
        'Posts_Per_Week': posts_per_week,
        'Top_Post_Product': top_post_entities['products'][0] if top_post_entities['products'] else 'N/A',
        'Top_Post_Price': top_post_entities['prices'][0] if top_post_entities['prices'] else 'N/A',
        'Avg_Price_ETB': avg_price,
        'Lending_Score': lending_score
    })

# Save scorecard
pd.DataFrame(vendor_metrics).to_csv('data/vendor_scorecard.csv', index=False)
