# lyrics-to-valence

**Research question:** How well can NLP models predict a song's emotional valence from its lyrics alone? Do lyrics actually capture how a song feels?

## Dataset

[Music Dataset: Lyrics and Metadata 1950–2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019) — 28,372 songs with full lyrics, metadata (artist, genre, release year), and Spotify audio features including a `valence` score (continuous 0–1 measure of musical positivity). We binarize `valence` at 0.5 to create `valence_label` (0 = low/sad, 1 = high/happy) for binary classification.

## Models

| Model  | Description |
|---|---|
| TF-IDF + Naive Bayes | Bag-of-words baseline with multinomial Naive Bayes |
| TF-IDF + Linear SVM | TF-IDF features fed into a linear support vector classifier |
| Fine-tuned RoBERTa | Transformer model fine-tuned end-to-end on lyrics sequences |

## Notebooks

| Notebook | Description |
|---|---|
| `00_preprocessing.ipynb` | Load raw CSV, clean lyrics, binarize valence, stratified 80/20 train/test split |
| `01_naive_bayes.ipynb` | TF-IDF vectorization + Multinomial Naive Bayes classifier |
| `02_svm.ipynb` | TF-IDF vectorization + Linear SVM classifier |
| `03_roberta.ipynb` | Fine-tuned RoBERTa transformer classifier |
| `04_evaluation.ipynb` | Side-by-side comparison of all models: accuracy, F1, ROC-AUC, confusion matrices |

## Setup

```bash
# 1. Clone the repo
git clone git@github.com:nidhijadhav/lyrics-to-valence.git
cd lyrics-to-valence

# 2. Install dependencies
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

3. Download the dataset from Kaggle and place the CSV at:
   ```
   data/tcc_ceds_music.csv
   ```
   The `data/` folder is tracked by git but all CSV files inside it are gitignored — the raw dataset and generated splits stay local only.

4. Run `00_preprocessing.ipynb` first. It reads the raw CSV and writes `data/train.csv` and `data/test.csv`, which all model notebooks depend on.

5. Run each model notebook (`01`, `02`, `03`) to train and evaluate. Each saves its test predictions to `data/<model>_preds.csv` for use by `04_evaluation.ipynb`.

6. The `results/` folder is tracked and is where final plots and output files should be saved. Notebooks currently save figures to `data/` during development; move any keeper plots to `results/` before committing.
