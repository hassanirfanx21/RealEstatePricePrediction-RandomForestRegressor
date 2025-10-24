# RealEstatePricePrediction-RandomForestRegressor

Simple house price prediction web app using a trained Random Forest regressor.

This project provides a minimal Flask web server (`server.py`) that loads:
- `Real-Estate-Price-Prediction.pickle` — trained model (expected to predict log(price)).
- `Bengaluru_House_Data.csv` — dataset used to compute location averages.
- `locations.json` — list of location names for the form dropdown.

What the app does
- A simple HTML form collects: BHK size, total sqft, bath, balcony, area type and location.
- The server converts inputs into the model feature vector and returns a formatted price.
- Front-end has a dark blue + maroon theme with improved dropdown readability.

How to run (Windows PowerShell)
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the server:

```powershell
python server.py
```

4. Open http://127.0.0.1:5000 in your browser.

Notes and troubleshooting
- If the model file or data files are missing, the app will print warnings and may not predict.
- If `Flask` is not installed, `pip install Flask` will fix it.
- GitHub pushes may require a Personal Access Token (PAT) instead of a password if you use HTTPS.

Files
- `server.py` — Flask server and inline HTML template.
- `Real-Estate-Price-Prediction.pickle` — the trained model file (binary).
- `Bengaluru_House_Data.csv` — data used to build location averages.
- `locations.json` — location dropdown source.
- `README.md` — this file.
- `requirements.txt` — pip dependencies.
- `.gitignore` — files to ignore when committing.

License
- Add a LICENSE file if you want to publish under a specific license.

Questions
- If you want a prettier standalone frontend, I can extract the template into separate HTML/CSS/JS files or add a client-side dropdown search.
