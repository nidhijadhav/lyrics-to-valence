# lyrics-to-valence

**Research question:** How well can NLP models predict a song's topic from its lyrics alone?

## Dataset

[Music Dataset: Lyrics and Metadata 1950–2019](https://data.mendeley.com/datasets/3t9vbwxgr5/2) - 28,372 songs with full lyrics, metadata (artist, genre, release year), and pre-labeled topic categories. We predict the `topic` column, which is an 8-class label, from cleaned lyrics alone.

**Topic classes** (integer-encoded alphabetically by `sklearn.LabelEncoder`):

| Label | Topic |
|---|---|
| 0 | feelings |
| 1 | music |
| 2 | night/time |
| 3 | obscene |
| 4 | romantic |
| 5 | sadness |
| 6 | violence |
| 7 | world/life |

## Models

| Model |  Description |
|---|---|
| TF-IDF + Naive Bayes | Bag-of-words baseline with Multinomial Naive Bayes |
| TF-IDF + Linear SVC | TF-IDF features with one-vs-rest Linear SVC |
| Fine-tuned RoBERTa | `roberta-base` fine-tuned end-to-end for 8-class classification (run in Colab) |

## Notebooks

| Notebook | Description |
|---|---|
| `00_preprocessing.ipynb` | Load raw CSV, clean lyrics, LabelEncode topics, stratified 80/20 split |
| `01_naive_bayes.ipynb` | TF-IDF + MultinomialNB — classification report, confusion matrix, top features per class |
| `02_svc.ipynb` | TF-IDF + LinearSVC — same metrics, accuracy per topic, top features per class |
| `03_roberta.ipynb` | Fine-tuned RoBERTa |
| `04_evaluation.ipynb` | Cross-model comparison: macro F1, per-class F1 bar chart, confusion matrices |

## Repo structure

```
lyrics-to-valence/
├── data/
│   ├── raw/                    # place dataset CSV here (gitignored)
│   ├── processed/              # train.csv, test.csv, topic_label_map.json
│   └── preds/                  # model prediction CSVs (nb_preds.csv, svm_preds.csv, roberta_preds.csv)
├── notebooks/
│   ├── 00_preprocessing.ipynb
│   ├── 01_naive_bayes.ipynb
│   ├── 02_svc.ipynb
│   ├── 03_roberta.ipynb
│   └── 04_evaluation.ipynb
├── results/                    # saved plots (tracked)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# 1. Clone the repo
git clone git@github.com:nidhijadhav/lyrics-to-valence.git
cd lyrics-to-valence

# 2. Install dependencies
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

3. Download the dataset from [here](https://data.mendeley.com/datasets/3t9vbwxgr5/2) and place the CSV at:
   ```
   data/raw/Music Dataset Lyrics and Metadata from 1950 to 2019 2/tcc_ceds_music.csv
   ```

4. Run `00_preprocessing.ipynb` first. It cleans lyrics, encodes the 8-class topic label, and writes `data/processed/train.csv`, `data/processed/test.csv`, and `data/processed/topic_label_map.json`.

5. Run `01_naive_bayes.ipynb` and `02_svc.ipynb` locally. Each saves predictions to `data/preds/<model>_preds.csv`.

6. Run `03_roberta.ipynb` in **Google Colab with a GPU runtime**. Download `roberta_preds.csv` from Colab and place it in `data/preds/`.

7. Run `04_evaluation.ipynb` to compare all three models side by side.
