# Hermes

Hermes is an academic NLP project that classifies tweet sentiment and figurative language through a simple Flask API.

The repository reflects an earlier stage of my work with machine learning and text pipelines. It is best read as a public learning project that combines preprocessing, classical ML, dataset handling and an API wrapper around prediction flows.

## What it does

- trains and serves Support Vector Machine based classifiers
- processes tweet-like text inputs for sentiment-oriented analysis
- exposes HTTP endpoints for keyword search and trend-related requests
- uses local configuration and cached resources to support the API flow

## Stack

- Python
- Flask
- scikit-learn
- pandas
- NLTK
- langdetect
- PyYAML
- snscrape

## Project structure

```text
main.py                 # Flask API and request flow
svm_algorithm.py        # training and prediction helpers
twitterwebcrawler.py    # collection helpers
translate_dataset.py    # translation utilities
datasets/               # local datasets used by the project
requirements.txt
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the project

Create and fill a `config.yml` file with the keys and dataset paths expected by `main.py`.

### 3. Run the API

```bash
python main.py
```

## Example endpoints

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"keyword":"potato"}' \
  http://hermesproject.pythonanywhere.com/search
```

```bash
curl -X GET \
  -H "X-API-Key: YOUR_KEY" \
  http://hermesproject.pythonanywhere.com/trends
```

## Dataset note

This repository references public tweet datasets such as Sentiment140 and may also contain locally assembled training material.

If you plan to reuse the datasets or the trained outputs, review provenance and licensing yourself before doing anything beyond personal study. This public repository is presented as a learning and portfolio project, not as a curated benchmark package.

## Why this repo stays in the portfolio

Hermes is still useful as a signal for:

- early NLP experimentation
- API exposure around ML logic
- classical ML workflow assembly
- practical interest in sentiment and text classification problems

## Contact

Public profile: [github.com/Mentorzx](https://github.com/Mentorzx)
