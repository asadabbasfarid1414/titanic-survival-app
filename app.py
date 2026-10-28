
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "titanic_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

st.subheader("Passenger Information")

with st.form("passenger_form"):
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        index=2
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )

    sibsp = st.number_input(
        "Siblings / Spouses aboard",
        min_value=0,
        max_value=10,
        value=0
    )

    parch = st.number_input(
        "Parents / Children aboard",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        }[x]
    )

    submitted = st.form_submit_button(
        "Predict Survival"
    )

if submitted:
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    prediction = model.predict(passenger)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(passenger)[0]
        st.write(
            f"Model-estimated survival probability: "
            f"{probabilities[1] * 100:.1f}%"
        )

st.caption(
    "Educational machine learning demo. "
    "Predictions are not historical certainties."
)
