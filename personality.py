import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load model & scaler

model = joblib.load("personality_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")


# Streamlit UI setup

st.set_page_config(page_title="AI Personality Predictor", page_icon="🧠", layout="centered")

st.title("🧠 AI Personality Prediction App")
st.write("This app predicts your personality type based on key behavioral and emotional traits. Rate yourself below (0–10 scale).")


# Input features

traits = {
    "Sociability": st.slider("Sociability (0 = reserved, 10 = outgoing)", 0, 10, 5),
    "Solo_preference": st.slider("Solo Preference (0 = hates alone time, 10 = enjoys solitude)", 0, 10, 5),
    "Talkative": st.slider("Talkativeness", 0, 10, 5),
    "Grouping": st.slider("Group Comfort", 0, 10, 5),
    "Partying": st.slider("Partying", 0, 10, 5),
    "Listening": st.slider("Listening Skill", 0, 10, 5),
    "Speaking": st.slider("Public Speaking Comfort", 0, 10, 5),
    "Curiosity": st.slider("Curiosity Level", 0, 10, 5),
    "Routine": st.slider("Routine Preference", 0, 10, 5),
    "Thrill": st.slider("Thrill/Excitement Seeking", 0, 10, 5),
    "Friendly": st.slider("Friendliness", 0, 10, 5),
    "Adventure": st.slider("Adventurousness", 0, 10, 5),
    "Stress": st.slider("Stress Handling", 0, 10, 5)
}

#Convert to array

input_data = np.array([list(traits.values())])
scaled_input = scaler.transform(input_data)


# Predict personality

if st.button("🔮 Predict Personality"):
    prediction = model.predict(scaled_input)[0]
    mapping = {0: "Ambivert", 1: "Extrovert", 2: "Introvert"}
    result = mapping.get(prediction, "Unknown")

    st.subheader(f"Your Predicted Personality: **{result}**")

    # Personality descriptions
    if result == "Extrovert":
        st.success("🎉 You are outgoing, energetic, and love social settings!")
    elif result == "Introvert":
        st.info("🌙 You prefer calm environments and enjoy deep reflection.")
    else:
        st.warning("⚖️ You balance both introversion and extroversion traits perfectly.")

    # Radar chart visualization
    st.write("### Your Personality Trait Profile:")
    categories = list(traits.keys())
    values = list(traits.values())

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax.plot(angles, values, color='cyan', linewidth=2)
    ax.fill(angles, values, color='cyan', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title(f"Personality Radar — {result}", size=14, color='white', pad=20)
    st.pyplot(fig)


# Footer

st.caption("Built by Uday Thakur 🧠 | Logistic Regression | Streamlit | Scikit-learn")