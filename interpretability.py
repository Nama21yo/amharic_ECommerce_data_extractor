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
