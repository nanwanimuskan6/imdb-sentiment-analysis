# Local model artifacts

Restore the trained IMDb DistilBERT export to artifacts/transformer/best_model/ from your own backup or request it from the project author. No hosted download is configured.

Required inference files: config.json, model.safetensors, tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.txt. Keep model and tokenizer from the same export. Base distilbert-base-uncased weights are not a substitute.

The best export is epoch 2 / checkpoint-2500. Training checkpoints also hold optimizer, scheduler, scaler, RNG and Trainer state; these are not needed for inference.

Keep local and out of normal Git commits:

- artifacts/transformer/best_model/ (weights approximately 255 MiB).
- artifacts/transformer/checkpoints/ (individual optimizer files exceed 500 MiB).
- Other generated Transformer artifacts except the small training configuration.
- artifacts/classical/ (vectorizer and classifier Joblib files).
- .cache/, .venv/, data exports, logs and raw error-analysis review exports.

Measured JSON reports, split membership and PNG figures remain eligible for Git. Nothing needs deleting locally. If weights are published later, use a model host or release distribution and add the real versioned link here.

The two experimental pretrained models download separately from Hugging Face on first use. Their cache is also ignored.
