# Models

Large trained artifacts are not stored in this repo (GitHub 100MB file limit):

- `saved_models/*/model.safetensors` (fine-tuned BERT, BioBERT, PubMedBERT)
- `saved_models/random_forest.pkl`, `rf_for_ensemble.pkl`, `svm.pkl`, `xgboost.json`
- `embeddings/*.npy`

Regenerate by running the notebooks in `../notebooks` in order (01 to 04).
Small files (tokenizers, configs, encoders, scalers, logistic regression, MLP) are included.
