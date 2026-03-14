# DDM — Dynamic Decision Model

A multi-label intent classifier for natural language queries. Given a user query, DDM predicts one or more intent labels (e.g. `general`, `content`, `vision`, `reminder`) using a TF-IDF + LinearSVC pipeline.

---

## Labels

| Label | Example Query |
|---|---|
| `general` | *"What is quantum mechanics?"* |
| `content` | *"Write a cover letter for a data analyst role"* |
| `generate image` | *"Generate image of a futuristic city at night"* |
| `google search` | *"Search cheapest flights to dubai on google"* |
| `reminder` | *"Remind me to take my medicine at 9pm"* |
| `system` | *"Turn off bluetooth"* |
| `vision` | *"What is written on this whiteboard?"* |

---

## How It Works

1. Queries are lowercased, stripped, and punctuation-removed via `preprocess()`
2. Text is vectorized using **TF-IDF** with unigrams and bigrams
3. Labels are encoded with `MultiLabelBinarizer` to support multi-label output
4. A `MultiOutputClassifier` wraps a calibrated **LinearSVC** — one binary classifier per label
5. At inference, `DDM(query)` returns a list of predicted labels

---

## Project Structure

```
├── ddm.py               # Main classifier script
├── training_data.json   # Training examples with query + labels
├── test_data.json       # Test examples for evaluation
└── README.md
```

---

## Setup

**Requirements:** Python 3.8+

```bash
pip install scikit-learn
```

---

## Usage

### Training & Evaluation

```bash
python ddm.py
```

This trains the model on `training_data.json`, runs evaluation on `test_data.json`, and prints a classification report with per-label precision, recall, and F1.

### Single Query Inference

```python
from ddm import DDM

DDM("remind me to call mom at 6pm")
# → ['reminder']

DDM("what is this image showing")
# → ['vision']
```

---

## Data Format

Both `training_data.json` and `test_data.json` follow the same schema:

```json
[
  { "query": "write a blog post about climate change", "labels": ["content"] },
  { "query": "take a screenshot",                      "labels": ["system"]  }
]
```

Multi-label entries are supported:

```json
{ "query": "search and generate image of a sunset", "labels": ["google search", "generate image"] }
```

---

## Model Details

| Component | Choice | Reason |
|---|---|---|
| Vectorizer | TF-IDF (1–2 grams) | Captures phrases like *"generate image"* or *"set a reminder"* |
| Classifier | LinearSVC | Fast and effective for high-dimensional sparse text data |
| Calibration | `CalibratedClassifierCV` | Enables probability estimates from LinearSVC |
| Multi-label | `MultiOutputClassifier` | Independent binary classifier per label |

---

## License

MIT