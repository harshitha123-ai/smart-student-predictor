from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Keep this order the same as the training notebook.
FEATURES = [
    ("study_hours", "Study Hours"),
    ("attendance_percent", "Attendance Percent"),
    ("assignment_marks", "Assignment Marks"),
    ("internal_marks", "Internal Marks"),
    ("previous_marks", "Previous Marks"),
    ("sleep_hours", "Sleep Hours"),
]
FEATURE_NAMES = [feature_name for _, feature_name in FEATURES]
MODEL_PATH = Path(__file__).parent / "student_performance_model.pkl"

# Keep the website open if the model file is missing or cannot be loaded.
model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)

    if not isinstance(model, RandomForestRegressor):
        raise ValueError(
            "The file does not contain a Random Forest regressor. "
            "Please use the existing Random Forest model file."
        )

    if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
        raise ValueError("The model does not expect exactly six input features.")

    saved_feature_names = getattr(model, "feature_names_in_", None)
    if saved_feature_names is not None and list(saved_feature_names) != FEATURE_NAMES:
        raise ValueError("The model feature names or feature order do not match the notebook.")
except Exception as error:
    model_error = str(error)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error_message = model_error
    entered_values = {}

    if request.method == "POST":
        entered_values = {
            form_name: request.form.get(form_name, "").strip()
            for form_name, _ in FEATURES
        }

        if model is None:
            error_message = f"The model could not be loaded: {model_error}"
        else:
            try:
                # Check each required value and convert it to a number.
                input_values = []
                for form_name, feature_name in FEATURES:
                    value = entered_values[form_name]
                    if not value:
                        raise ValueError(f"Please enter a value for {feature_name}.")
                    try:
                        input_values.append(float(value))
                    except ValueError:
                        raise ValueError(f"{feature_name} must be a number.") from None

                # A DataFrame preserves the exact names and order used during training.
                input_data = pd.DataFrame([input_values], columns=FEATURE_NAMES)
                prediction = float(model.predict(input_data)[0])
                error_message = None
            except ValueError as error:
                error_message = str(error)
            except Exception:
                error_message = "Prediction failed. Check the input values and model file."

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        error_message=error_message,
        entered_values=entered_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
    
