from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
import time
from seqeval.metrics import classification_report

# Load CoNLL data (same as fine_tune_ner.py)
def load_conll(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                token, label = line.strip().split()
                current_sentence.append(token)
                current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    return pd.DataFrame({'tokens': sentences, 'ner_tags': labels})

# Tokenize and align labels
label_list = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith('I-') else -100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Evaluate inference speed
def measure_inference_speed(model, tokenizer, dataset):
    start_time = time.time()
    predictions = []
    for example in dataset:
        inputs = tokenizer(example['tokens'], is_split_into_words=True, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        predictions.append(outputs.logits.argmax(dim=-1).tolist())
    return (time.time() - start_time) / len(dataset)

# Fine-tune and evaluate models
models = [
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
    "bert-base-multilingual-cased"
]
results = []

data = load_conll('data/labeled_conll.txt')
dataset = Dataset.from_pandas(data)
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test['train']
eval_dataset = train_test['test']

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)['train']
    eval_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)['test']
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.split('/')[-1]}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    inference_time = measure_inference_speed(model, tokenizer, eval_dataset)
    
    results.append({
        'model': model_name,
        'f1_score': eval_results['eval_f1'] if 'eval_f1' in eval_results else 'N/A',
        'inference_time_per_sample': inference_time,
        'robustness': 'High' if 'xlm-roberta' in model_name else 'Moderate'
    })

# Save comparison results
pd.DataFrame(results).to_csv('data/model_comparison.csv', index=False)
print(pd.DataFrame(results))
