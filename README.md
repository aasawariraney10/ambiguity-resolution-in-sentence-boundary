# Ambiguity Resolution in Sentence Boundary in Natural Language Text

This project focuses on resolving ambiguities in sentence boundary detection within natural language text. Sentence boundary disambiguation is a critical task in Natural Language Processing (NLP), especially when dealing with unstructured text that may contain abbreviations, decimal numbers, or irregular punctuation patterns.

## 📌 Objective

To build and evaluate a deep learning model that can accurately classify and predict sentence boundaries, even when ambiguities are present.

## 🧠 Problem Context

Traditional rule-based sentence splitters often fail in the presence of:
- Abbreviations (e.g., "Dr.", "etc.")
- Decimal points (e.g., "3.14")
- Misused punctuation (e.g., "What?!")

This project leverages an LSTM-based neural network to learn and predict sentence boundary locations using contextual patterns in labeled examples.

## 📂 Project Structure

- `ambiguity_resolution_in_sentence_boundary.ipynb` – Jupyter Notebook with full code for data loading, model training, evaluation, and prediction.
- `Major Project Dataset.xlsx` – Dataset containing labeled text (`X`, `Y`) for training the model.
- `NLP_Prediction_Data.xlsx` – Test data used to evaluate generalization of the trained model.
- `README.md` – Project overview and instructions.

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow/Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib
- Google Colab (for development and training)

## 📈 Model Architecture

- **Tokenizer & Padding:** Converts raw text into padded sequences.
- **Embedding Layer:** Transforms input into dense vectors.
- **Bidirectional LSTM:** Learns temporal patterns from both past and future contexts.
- **Dense Softmax Layer:** Outputs multi-class predictions for sentence boundary status.

## 📊 Results & Metrics

The model is evaluated using precision, recall, and F1-score, with a focus on how well it handles edge cases and ambiguity.

To run evaluation:
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, zero_division=0))
