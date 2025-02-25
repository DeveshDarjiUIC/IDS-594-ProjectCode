from flask import Flask, request, jsonify, render_template_string
import skops.io as sio
import pandas as pd
from io import StringIO

app = Flask(__name__)

# Load your saved model (ensure 'bank_model.skops' is accessible)
model = sio.load("bank_model.skops")

# A basic HTML template with a text area for CSV data input.
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Bank Model Prediction</title>
</head>
<body>
    <h1>Bank Model Prediction</h1>
    <p>Please paste your CSV data below. Make sure your CSV includes a header row with feature names.</p>
    <form action="/predict" method="post">
        <textarea name="csv_data" rows="10" cols="80" placeholder="feature1,feature2,feature3\n0.5,1.2,3.4"></textarea><br>
        <input type="submit" value="Predict">
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(html_template)

@app.route("/predict", methods=["POST"])
def predict():
    csv_data = request.form.get("csv_data")
    try:
        # Convert the pasted CSV string into a DataFrame using StringIO
        data_df = pd.read_csv(StringIO(csv_data))
        
        # IMPORTANT: Ensure that data_df has the same features/columns expected by your model.
        predictions = model.predict(data_df)
        
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
