"""
models/
=======
ShopSmart trained-model artefacts.

After running ``python models/train_model.py`` two files appear here:
  model.pkl      – fitted sklearn Pipeline (joblib, compress=3)
  metrics.json   – test/CV metrics + config

Quick-load:
    import joblib
    pipeline = joblib.load("models/model.pkl")
    preds = pipeline.predict(X_new)
"""
