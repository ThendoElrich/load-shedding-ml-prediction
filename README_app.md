# Running the Load-Shedding Predictor app

1. pip install -r requirements.txt
2. In one terminal:  python api.py
   (starts the Flask API on http://127.0.0.1:5000)
3. In a second terminal:  streamlit run app.py
   (opens the web form in your browser)

Both files, plus final_model.pkl, scaler.pkl, feature_names.json, and
threshold.json (all already saved by this notebook), must sit in the
same folder for the API to load correctly.
