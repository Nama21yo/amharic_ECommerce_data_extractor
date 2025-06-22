import pandas as pd

def label_message(tokens):
    # Dummy labeling function (in practice, use manual annotation or a tool)
    labels = []
    for token in tokens:
        if token in ['3pcs', 'silicon', 'brush', 'spatulas']:
            if token == '3pcs':
                labels.append('B-PRODUCT')
            else:
                labels.append('I-PRODUCT')
        elif token.startswith('ዋጋ') or token.endswith('ብር'):
            if token.startswith('ዋጋ'):
                labels.append('B-PRICE')
            else:
                labels.append('I-PRICE')
        elif token in ['አዲስ', 'አበባ', 'ቦሌ']:
            if token == 'አሉ':
                labels.append('B-LOC')
            else:
                labels.append('I-LOC')
        else:
            labels.append('O')
    return labels

def generate_conll():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv', encoding='utf-8')
    
    # Sample 30 messages (for demo)
    sample_df = df.sample(n=30, random_state=42)
    
    # Generate CoNLL output
    with open('data/labeled_conll.txt', 'w', encoding='utf-8') as f:
        for _, row in sample_df.iterrows():
            tokens = eval(row['tokens'])  # Convert string to list
            labels = label_message(tokens)
            for token, label in zip(tokens, labels):
                f.write(f"{token} {label}\n")
            f.write('\n')

if __name__ == '__main__':
    generate_conll()
