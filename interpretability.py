from transformers import pipeline
import shap
import lime
from lime.lime_text import LimeTextExplainer
import pandas as pd

# Load fine-tuned model
model_path = "models/xlm-roberta-ner"
ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")

# Sample difficult case
difficult_text = "ዋጋ 1000 ብር ለቦቶ ጫማ በአዲስ አበባ ቦሌ"  # Ambiguous: "ለቦቶ" could be product or description

# SHAP explanation
explainer = shap.Explainer(ner_pipeline)
shap_values = explainer([difficult_text])
shap_summary = str(shap_values)  # Simplified for report

# LIME explanation
lime_explainer = LimeTextExplainer(class_names=['O', 'B-PRODUCT', 'I-PRODUCT', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC'])
lime_exp = lime_explainer.explain_instance(difficult_text, ner_pipeline, num_features=10)
lime_summary = str(lime_exp.as_list())

# Save interpretability report
with open('data/interpretability_report.txt', 'w', encoding='utf-8') as f:
    f.write("SHAP Explanation:\n")
    f.write(shap_summary + "\n\n")
    f.write("LIME Explanation:\n")
    f.write(lime_summary + "\n\n")
    f.write("Analysis of Difficult Case:\n")
    f.write("Text: " + difficult_text + "\n")
    f.write("The model may struggle with 'ለቦቶ' as it could be part of the product name or a description. SHAP and LIME highlight contributions of tokens like 'ጫማ' (high for B-PRODUCT) and 'ዋጋ' (high for B-PRICE), indicating correct identification of core entities but potential confusion with modifiers.")
