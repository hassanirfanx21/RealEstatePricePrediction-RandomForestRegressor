from flask import Flask, request, render_template_string, flash
import json
import numpy as np
import pandas as pd
import pickle
import traceback

app = Flask(__name__)

# HERE I LOAD DATA FILES AND MODEL(which is trained and saved as pickle file)
try:
    with open(
        "locations.json", "r"
    ) as f:  # here i import location .json file which contains list of locations
        locations = [
            x.strip() for x in json.load(f)
        ]  # Normalize locations and gets list from locations.json
    print(f" Loaded {len(locations)} locations.")  # tells how many locations loaded
except Exception as e:
    print("Could not load locations.json:!!!!", e)
    locations = []
# here we read the csv to use for dynamic location encoding by "Mean Encoding"
try:
    df = pd.read_csv("Bengaluru_House_Data.csv")
    df["location"] = (
        df["location"].astype(str).str.strip().str.lower()
    )  # Normalize location names
    location_price_map = (
        df.groupby("location")["price"].mean().to_dict()
    )  # Create mean price mapping and store in dict
    global_mean_price = df[
        "price"
    ].mean()  # Get global mean price for unknown locations
    print(f"Loaded {len(location_price_map)} location encodings.")
except Exception as e:
    print(" Could not load Bengaluru_House_Data.csv:!!!!!", e)
    location_price_map = {}  # empty dict
    global_mean_price = 0  # Get global mean price for unknown locations

try:  # load the trained model, as i stored in pickle file after training, it uses sklearn inside so import it too
    with open("Real-Estate-Price-Prediction.pickle", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Failed to load model:", e)
    traceback.print_exc()

# -------------------------------------------------------
# HTML TEMPLATE
# -------------------------------------------------------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>House Price Predictor</title>
    <style>
        :root {
            --bg1: #0a1930; /* dark navy */
            --bg2: #5a0f2f; /* maroon */
            --card-bg: rgba(10, 15, 25, 0.65);
            --accent: #5cc8ff; /* cyan accent */
            --accent-2: #ff6b6b; /* coral for highlights */
            --text: #e6eefc;
            --muted: #9fb3c8;
            --input-bg: rgba(255,255,255,0.06);
            --input-border: rgba(255,255,255,0.12);
            --success: #2ed573;
            --error: #ff6b6b;
        }
        * { box-sizing: border-box; }
        html, body { height: 100%; }
        body {
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            color: var(--text);
            background:
                radial-gradient(1200px 800px at 20% 10%, rgba(92,200,255,0.06), transparent 60%),
                radial-gradient(1000px 700px at 80% 90%, rgba(255,107,107,0.07), transparent 60%),
                linear-gradient(135deg, var(--bg1), var(--bg2));
        }
        .container {
            min-height: 100%;
            display: grid;
            place-items: center;
            padding: 32px 16px;
        }
        .card {
            width: min(720px, 92vw);
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
            padding: 28px;
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: "";
            position: absolute;
            inset: -2px;
            background: conic-gradient(from 180deg at 50% 50%, rgba(92,200,255,0.15), rgba(255,107,107,0.15), rgba(92,200,255,0.15));
            filter: blur(26px);
            z-index: -1;
        }
        .header { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
        .logo {
            width: 40px; height: 40px; display: grid; place-items: center;
            background: linear-gradient(135deg, rgba(92,200,255,0.25), rgba(255,107,107,0.25));
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px; color: var(--text); font-weight: 700; letter-spacing: 0.5px;
        }
        h1 { font-size: clamp(20px, 3vw, 26px); margin: 0; font-weight: 700; }
        .sub { margin: 0; color: var(--muted); font-size: 14px; }
        hr.sep { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 18px 0 22px; }
        .alerts { margin: 0 0 12px; padding: 0; list-style: none; }
        .alerts li {
            background: rgba(255,107,107,0.12);
            border: 1px solid rgba(255,107,107,0.35);
            color: var(--error);
            padding: 10px 12px; border-radius: 10px; margin: 8px 0;
        }
        .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 14px 16px; }
        .form-grid .full { grid-column: 1 / -1; }
        .label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
            .input, select {
            width: 100%; padding: 12px 12px; background: var(--input-bg); color: var(--text);
            border: 1px solid var(--input-border); border-radius: 10px; outline: none;
            transition: border-color .2s, box-shadow .2s; appearance: none;
        }
        .input:focus, select:focus { border-color: rgba(92,200,255,0.7); box-shadow: 0 0 0 4px rgba(92,200,255,0.15); }
            /* Ensure dropdown list is dark with white text */
            select { background-color: #0e1a2f; color: #e6eefc; border-color: rgba(255,255,255,0.18); }
            select option { background-color: #14233a; color: #ffffff; }
            select option:hover { background-color: #1e3458; color: #ffffff; }
            select option:checked { background-color: #1b4b9b; color: #ffffff; }
            optgroup { color: #cbd6ea; background-color: #14233a; }
        .actions { display: flex; gap: 12px; margin-top: 10px; }
        button { cursor: pointer; border: 0; border-radius: 12px; padding: 12px 16px; font-weight: 600; letter-spacing: 0.3px; transition: transform .05s ease, filter .2s ease, box-shadow .2s ease; }
        .btn-primary {
            background: linear-gradient(135deg, #1b4b9b, #0d2a57);
            color: #eef6ff; box-shadow: 0 8px 20px rgba(27,75,155,0.35);
            border: 1px solid rgba(92,200,255,0.3);
        }
        .btn-primary:hover { filter: brightness(1.05); box-shadow: 0 10px 24px rgba(27,75,155,0.45); }
        .btn-primary:active { transform: translateY(1px); }
        .result {
            margin-top: 18px; background: linear-gradient(180deg, rgba(46,213,115,0.12), rgba(46,213,115,0.06));
            border: 1px solid rgba(46,213,115,0.35); color: var(--text);
            border-radius: 12px; padding: 14px 16px;
            display: flex; align-items: center; gap: 10px;
        }
        .result .badge {
            background: rgba(46,213,115,0.18); color: #c9ffdd;
            border: 1px solid rgba(46,213,115,0.45);
            padding: 6px 10px; border-radius: 999px; font-size: 12px;
            font-weight: 700; letter-spacing: .4px;
        }
        .price {
            font-weight: 800; font-size: clamp(18px, 3.4vw, 24px); color: #c9ffdd;
            text-shadow: 0 0 18px rgba(46,213,115,0.35);
        }
        @media (max-width: 560px) { .form-grid { grid-template-columns: 1fr; } .actions { flex-direction: column; } }
    </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="header">
                    <div class="logo">HP</div>
                    <div>
                        <h1>House Price Predictor</h1>
                        <p class="sub">User-friendly interface for predicting house prices</p>
                    </div>
                </div>
                <hr class="sep" />

                {% with messages = get_flashed_messages() %}
                    {% if messages %}
                        <ul class="alerts">
                            {% for m in messages %}<li>{{ m }}</li>{% endfor %}
                        </ul>
                    {% endif %}
                {% endwith %}

                <form method="post">
                    <div class="form-grid">
                        <div>
                            <label class="label">Size (e.g. 2 for 2 BHK)</label>
                            <input class="input" type="number" step="any" name="size" value="{{ form.get('size','') }}" required>
                        </div>
                        <div>
                            <label class="label">Total sqft</label>
                            <input class="input" type="number" step="any" name="total_sqft" value="{{ form.get('total_sqft','') }}" required>
                        </div>
                        <div>
                            <label class="label">Bath</label>
                            <input class="input" type="number" step="any" name="bath" value="{{ form.get('bath','') }}" required>
                        </div>
                        <div>
                            <label class="label">Balcony</label>
                            <input class="input" type="number" step="any" name="balcony" value="{{ form.get('balcony','') }}" required>
                        </div>
                        <div>
                            <label class="label">Area Type</label>
                            <select name="area_type" required>
                                <option value="Carpet Area" {% if form.get('area_type') == 'Carpet Area' %}selected{% endif %}>Carpet Area</option>
                                <option value="Plot Area" {% if form.get('area_type') == 'Plot Area' %}selected{% endif %}>Plot Area</option>
                                <option value="Super built-up Area" {% if form.get('area_type') == 'Super built-up Area' %}selected{% endif %}>Super built-up Area</option>
                                <option value="Built-up Area" {% if form.get('area_type') == 'Built-up Area' %}selected{% endif %}>Built-up Area</option>
                            </select>
                        </div>
                        <div>
                            <label class="label">Location</label>
                            <select name="location" required>
                                <option value="">-- choose location --</option>
                                {% for loc in locations %}
                                    <option value="{{ loc }}" {% if form.get('location') == loc %}selected{% endif %}>{{ loc }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="full actions">
                            <button class="btn-primary" type="submit">Predict Price</button>
                        </div>
                    </div>
                </form>

                {% if result %}
                    <div class="result">
                        <span class="badge">Result</span>
                        <span class="price">Predicted price: {{ result }}</span>
                    </div>
                {% endif %}
            </div>
        </div>
    </body>
    </html>
"""


# -------------------------------------------------------
# HELPER FUNCTION
# -------------------------------------------------------
# as in trained model the area  type is one hot encoded, we need to convert the input area type to one hot encoding
def area_to_onehot(area_type):
    t = area_type.lower()
    if t == "carpet area":
        return (1, 0, 0)
    if t == "plot area":
        return (0, 1, 0)
    if t.startswith("super built-up"):
        return (0, 0, 1)
    return (0, 0, 0)


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route("/", methods=["GET", "POST"])  # main page route
def index():
    if request.method == "POST":
        # Capture raw form values so we can re-fill the form after submit
        form_data = {
            "size": request.form.get("size", ""),
            "total_sqft": request.form.get("total_sqft", ""),
            "bath": request.form.get("bath", ""),
            "balcony": request.form.get("balcony", ""),
            "area_type": request.form.get("area_type", ""),
            "location": request.form.get("location", ""),
        }

        if model is None:# this will run if the model is not loaded correctly and not exist
            flash("❌ Model not loaded.")
            return render_template_string(
                TEMPLATE, locations=locations, result=None, form=form_data
            )

        try:  # parse numeric inputs from form because form inputs are strings, so we need to convert them to float
            size = float(form_data["size"]) if form_data["size"] != "" else float("nan")
            sqft = (
                float(form_data["total_sqft"])
                if form_data["total_sqft"] != ""
                else float("nan")
            )
            bath = float(form_data["bath"]) if form_data["bath"] != "" else float("nan")
            balcony = (
                float(form_data["balcony"])
                if form_data["balcony"] != ""
                else float("nan")
            )
        except Exception:
            flash("Please enter valid numbers.")
            return render_template_string(
                TEMPLATE, locations=locations, result=None, form=form_data
            )
        # other inputs
        area_type = form_data["area_type"]
        location = form_data["location"].strip().lower()

        carpet, plot, superbuilt = area_to_onehot(
            area_type
        )  # get one hot encoding for area type
        location_encoded = location_price_map.get(
            location, global_mean_price
        )  # get location encoding, or global mean if unknown
        # prepare feature array for model to predict
        features = np.array(
            [[size, sqft, bath, balcony, carpet, plot, superbuilt, location_encoded]]
        )

        try:
            log_price = model.predict(features)[0]
            price = np.expm1(log_price)
            result = f"₹ {price:,.2f}"
        except Exception as e:
            traceback.print_exc()
            flash("❌ Prediction failed.")
            result = None

        return render_template_string(
            TEMPLATE, locations=locations, result=result, form=form_data
        )

    # Initial GET render with empty form
    return render_template_string(TEMPLATE, locations=locations, result=None, form={})


# -------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
