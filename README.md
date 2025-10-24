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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
