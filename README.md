# Hybrid AI Model for Detecting Fake News Using NLP and Machine Learning


A hybrid voting ensemble (Naive Bayes + SVM + Random Forest) built on TF-IDF features to classify news articles as **fake** or **real**. The models are trained on the ISOT dataset and then tested on the LIAR dataset to measure how well they generalise to unseen data.

---

## Key Findings

- All tree/linear models reach **~99.4–99.6% accuracy** on the ISOT test set.
- Accuracy drops to **49–55%** on the LIAR dataset, so the models do not generalise beyond ISOT.
- The word `reuters` is the single most influential feature, which suggests the models partly learn **source identity** rather than purely fake-news characteristics.

---

## Results

### ISOT test set

| Model | Accuracy | F1 (weighted) | AUC |
|---|---|---|---|
| Naive Bayes | 0.9495 | 0.9496 | 0.9892 |
| SVM (LinearSVC) | 0.9963 | 0.9963 | 0.9996 |
| Random Forest | 0.9963 | 0.9963 | 0.9997 |
| **Hybrid Ensemble** | 0.9940 | 0.9940 | 0.9995 |

![Model comparison]()

### 5-fold cross-validation (training set)

| Model | Mean F1 | Std |
|---|---|---|
| Naive Bayes | 0.9556 | ±0.0027 |
| SVM | 0.9956 | ±0.0003 |
| Random Forest | 0.9963 | ±0.0007 |
| Hybrid Ensemble | 0.9953 | ±0.0004 |

### ROC curves

![ROC curves]()

### Confusion matrices

| SVM | Random Forest |
|---|---|
| ![SVM]() | ![Random Forest]() |

| Naive Bayes | Hybrid Ensemble |
|---|---|
| ![Naive Bayes]() | ![Ensemble]() |

### Feature importance

![Random Forest features]()
![SVM coefficients]()

### Cross-dataset generalisation (ISOT-trained, tested on LIAR)

| Model | ISOT accuracy | LIAR accuracy |
|---|---|---|
| Naive Bayes | 0.9495 | 0.49 |
| SVM | 0.9963 | 0.55 |
| Random Forest | 0.9963 | 0.55 |
| Hybrid Ensemble | 0.9940 | 0.55 |

![ISOT vs LIAR]()

**Why the drop?** ISOT contains full-length articles (real news comes from Reuters), while LIAR contains short political claims. The vocabulary and style differ, and the models rely on source-specific cues such as `reuters`.

---

## Methodology

1. **Data:** ISOT Fake News Dataset (44,898 articles; 23,481 fake, 21,417 real)
2. **Preprocessing:** lowercasing, URL/special-character removal, stop-word removal, lemmatisation, short-article filtering
3. **Features:** TF-IDF (unigrams + bigrams, sublinear TF)
4. **Models:** Naive Bayes, Linear SVM, Random Forest (200 trees), soft-voting ensemble
5. **Evaluation:** 80/20 stratified split, 5-fold cross-validation, classification report, confusion matrix, ROC/AUC, feature importance
6. **External validation:** LIAR test set (binary labels; "half-true" removed)

---

## Project Structure

```
fake_news_detection/
├── data/
│   ├── isot/
│   │   ├── Fake.csv
│   │   └── True.csv
│   ├── liar/
│   │   └── test.tsv
│   └── processed/            
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── models/                                    
├── app.py                    # Streamlit demo app
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Use | Link |
|---|---|---|
| ISOT Fake News | Training and testing | https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset |
| LIAR | External validation (`test.tsv` only) | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |

Place `Fake.csv` and `True.csv` in `data/isot/`, and `test.tsv` in `data/liar/`.

---

## Setup

```bash
git clone https://github.com/<your-username>/fake_news_detection.git
cd fake_news_detection
pip install -r requirements.txt
```

Then run the notebooks in order, from `01` to `05`.

## Run the web app

After running the notebooks (they save the models into `models/`):

```bash
streamlit run app.py
```

Paste a news article and the app returns the ensemble verdict, each model's prediction, and confidence scores.

---

## Limitations

- ISOT real news comes exclusively from Reuters, so models can learn the publisher instead of the content.
- Performance does not transfer to short claims (LIAR).
- The system is intended for academic demonstration, not as the sole basis for judging news authenticity.

## Hardware

CPU is sufficient for all models. Google Colab (free tier) works if your machine is slow.