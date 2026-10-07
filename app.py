import streamlit as st
from pathlib import Path
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Model path
MODEL_PATH = Path(__file__).parent / "student_performance_model.pkl"

FEATURES = [
    ("study_hours", "Study Hours"),
    ("attendance_percent", "Attendance Percent"),
    ("assignment_marks", "Assignment Marks"),
    ("internal_marks", "Internal Marks"),
    ("previous_marks", "Previous Marks"),
    ("sleep_hours", "Sleep Hours"),
]

FEATURE_NAMES = [feature_name for _, feature_name in FEATURES]

# Load model
try:
    model = joblib.load(MODEL_PATH)
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)


# Header
st.title("🎓 Student Performance Predictor")
st.write("AI-powered academic performance insights using Random Forest Regression.")
st.caption("Machine Learning • Random Forest Regression")

st.divider()

if model_error:
    st.error("Model could not be loaded.")
    st.code(model_error)
else:

    # Input section
    st.subheader("📊 Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "📚 Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        attendance = st.number_input(
            "📝 Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        assignment_marks = st.number_input(
            "📄 Assignment Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    with col2:
        internal_marks = st.number_input(
            "📖 Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        previous_marks = st.number_input(
            "🎯 Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        sleep_hours = st.number_input(
            "😴 Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    st.divider()

    # Buttons
    predict_col, reset_col = st.columns(2)

    with predict_col:
        predict = st.button(
            "🚀 Predict Performance",
            use_container_width=True
        )

    with reset_col:
        reset = st.button(
            "🔄 Reset",
            use_container_width=True
        )

    if reset:
        st.rerun()

    # Prediction
    if predict:

        input_values = [
            study_hours,
            attendance,
            assignment_marks,
            internal_marks,
            previous_marks,
            sleep_hours
        ]

        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURE_NAMES
        )

        try:
            prediction = float(model.predict(input_data)[0])

            # Keep score between 0 and 100
            prediction = max(0, min(100, prediction))

            if prediction >= 85:
                performance = "Excellent 🌟"
                message = "Outstanding performance! Keep maintaining your consistency."
            elif prediction >= 70:
                performance = "Good 👍"
                message = "Good performance. With a little more improvement, you can reach excellent levels."
            elif prediction >= 50:
                performance = "Average 📈"
                message = "There is good scope for improvement. Focus on study time and consistency."
            else:
                performance = "Needs Improvement 💪"
                message = "Increase your study consistency and focus on improving academic performance."

            # Result
            st.success("Prediction completed successfully!")

            st.subheader("🎯 Prediction Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Predicted Score",
                    f"{prediction:.2f}/100"
                )

            with r2:
                st.metric(
                    "Performance Level",
                    performance
                )

            with r3:
                st.metric(
                    "Attendance",
                    f"{attendance:.0f}%"
                )

            st.progress(int(prediction))

            st.info(message)

            # Input-based insights
            st.subheader("💡 Student Insights")

            insights = []

            if study_hours < 4:
                insights.append("📚 Try increasing your daily study hours.")
            else:
                insights.append("📚 Your study hours are at a reasonable level.")

            if attendance < 75:
                insights.append("📝 Improving attendance may help maintain academic consistency.")
            else:
                insights.append("📝 Your attendance is good.")

            if sleep_hours < 6:
                insights.append("😴 Try maintaining healthier sleep hours for better concentration.")
            elif sleep_hours >= 7:
                insights.append("😴 Your sleep duration is supportive of regular study.")

            if previous_marks < 60:
                insights.append("🎯 Focus on strengthening your previous weak areas.")
            else:
                insights.append("🎯 Your previous academic performance is a good foundation.")

            for insight in insights:
                st.write(insight)

            # Target
            st.subheader("🚀 Next Target")

            target = min(100, prediction + 10)

            st.write(
                f"Your next target can be **{target:.0f}/100**. "
                "Focus on consistent study, attendance and assignments."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.write(str(error))


st.divider()

st.caption(
    "Student Performance Prediction | Machine Learning | B.Tech CSE"
)
