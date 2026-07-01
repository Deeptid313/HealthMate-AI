# Text Feature Engineering Assignment

This repository contains a complete notebook-based workflow for a text feature engineering built on real Amazon review data. The project starts from raw review scraping, moves through preprocessing and vocabulary creation, then implements `One Hot Encoding`, `Bag of Words`, `TF-IDF`, sparse-matrix analysis, and a small sentiment-classification use case.

## What This Repo Is About

The goal of the project is to take real user-generated review text and convert it into machine-learning-ready numerical features. The work in this repository follows the assignment flow end to end:

1. Scrape product reviews and store them in CSV format.
2. Clean and preprocess the text.
3. Build a reusable vocabulary.
4. Generate text feature representations.
5. Compare OHE, BoW, and TF-IDF.
6. Analyze sparsity.
7. Run a mini sentiment-classification use case.

## What Has Been Implemented

### 1. Dataset Collection

Notebook: `webscapping.ipynb`

- Scraped Amazon product reviews.
- Stored the collected reviews in `data/amazon_reviews.csv`.
- Final raw dataset size: `690` reviews.

### 2. Preprocessing

Notebook: `preprocessing.ipynb`

Implemented preprocessing steps:

- lowercase conversion
- tokenization using NLTK
- punctuation removal
- stopword removal
- lemmatization
- export of cleaned text

The notebook also cleans scraping artifacts such as trailing `Read more` text and spacing issues like merged words.

Output:

- `data/amazon_reviews_preprocessed.csv`

Final cleaned dataset size:

- `688` usable reviews

### 3. Vocabulary Creation

Notebook: `vocabulary.ipynb`

Built a reusable vocabulary from the preprocessed review text using `CountVectorizer`.

Outputs:

- `data/amazon_reviews_vocabulary.csv`
- `data/amazon_reviews_vocabulary_mapping.json`

Vocabulary size:

- `3251` unique terms

Top frequent words in the corpus:

- `good`
- `quality`
- `sound`
- `battery`
- `earbuds`

### 4. Feature Engineering

Notebook: `featureengineering.ipynb`

Implemented:

- One Hot Encoding
- Bag of Words
- TF-IDF
- comparison analysis
- sparse matrix analysis
- short written answers to the theory questions

Feature matrix shape for all three methods:

- `688 x 3251`

Sparsity for all three matrices:

- `98.6795%`

This shows that each review uses only a small fraction of the full vocabulary, which is typical for text data.

### 5. Mini Use Case: Sentiment Classification

Notebook: `usecase.ipynb`

Implemented a simple binary sentiment classification task using the review ratings:

- `4–5 stars` -> `positive`
- `1–2 stars` -> `negative`
- `3 stars` -> removed as neutral

Class distribution used for the use case:

- `555` positive reviews
- `69` negative reviews

Model used:

- `Logistic Regression`

Feature comparison:

- `BoW` accuracy: `0.968`
- `TF-IDF` accuracy: `0.976`

Best result on the current split:

- `TF-IDF`

## Repository Structure

```text
.
├── data/
│   ├── amazon_reviews.csv
│   ├── amazon_reviews_preprocessed.csv
│   ├── amazon_reviews_vocabulary.csv
│   └── amazon_reviews_vocabulary_mapping.json
├── webscapping.ipynb
├── preprocessing.ipynb
├── vocabulary.ipynb
├── featureengineering.ipynb
├── usecase.ipynb
├── pyproject.toml
└── README.md
```

## Tech Stack

- Python `3.12`
- `pandas`
- `nltk`
- `scikit-learn`
- `requests`
- `beautifulsoup4`
- `ipykernel`

## How To Run

Install dependencies in the project environment, then run the notebooks in this order:

1. `webscapping.ipynb`
2. `preprocessing.ipynb`
3. `vocabulary.ipynb`
4. `featureengineering.ipynb`
5. `usecase.ipynb`

If the cleaned dataset and exported vocabulary files already exist, you can start from:

- `preprocessing.ipynb` to rebuild cleaned text
- `vocabulary.ipynb` to rebuild vocabulary artifacts
- `featureengineering.ipynb` and `usecase.ipynb` to continue the later tasks

## Key Outcomes

- Built a complete text-processing pipeline from raw reviews to model-ready features.
- Exported reusable cleaned data and vocabulary artifacts for downstream tasks.
- Compared three standard feature engineering approaches on the same corpus.
- Measured matrix sparsity and showed why text representations are high-dimensional.
- Demonstrated a simple review-sentiment classifier, where `TF-IDF` slightly outperformed `BoW` on the current split.

## Notes

- This project is notebook-first and demonstration.
- The use-case labels are derived from review star ratings, so they are a practical proxy for sentiment rather than manually annotated sentiment labels.
