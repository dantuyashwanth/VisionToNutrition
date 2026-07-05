import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf

from dotenv import dotenv_values, load_dotenv
from google import genai
from google.genai import types

from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications.efficientnet import preprocess_input


# ==========================================================
# APP CONFIG
# ==========================================================

APP_TITLE = "🍽️ AI Indian Food Nutrition Assistant"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_efficientnet.keras"

CLASS_FILE = BASE_DIR / "class_names.json"

GEMINI_MODEL = "gemini-2.5-flash"


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_food_model():
    return tf.keras.models.load_model(MODEL_PATH)


food_model = load_food_model()


# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

@st.cache_data
def load_classes():

    with open(CLASS_FILE, "r") as f:

        return json.load(f)


CLASS_NAMES = load_classes()


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(
        uploaded_file,
        max_size=(1024, 1024)
):

    if uploaded_file is None:

        raise ValueError(
            "Please upload an image."
        )

    try:

        image = Image.open(uploaded_file)

        image.verify()

        uploaded_file.seek(0)

        image = Image.open(uploaded_file).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise ValueError(
            "Invalid image."
        )

    image.thumbnail(max_size)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    return image, buffer.getvalue()


# ==========================================================
# CNN PREDICTION
# ==========================================================

def predict_food_category(image):

    img = image.resize((224,224))

    img = np.array(img).astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = food_model.predict(
        img,
        verbose=0
    )[0]

    top5 = prediction.argsort()[-5:][::-1]

    results = []

    for idx in top5:

        results.append({

            "food": CLASS_NAMES[idx],

            "confidence": float(
                prediction[idx]*100
            )

        })

    return results
# ==========================================================
# BMI FUNCTIONS
# ==========================================================

def calculate_bmi(weight_kg, height_m):

    if height_m <= 0:
        raise ValueError("Height must be greater than 0")

    return weight_kg / (height_m ** 2)


def classify_bmi(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obese"


# ==========================================================
# GEMINI API
# ==========================================================

def get_api_key():

    env_path = BASE_DIR / ".env"

    load_dotenv(env_path)

    key = ""

    if env_path.exists():

        key = dotenv_values(
            env_path
        ).get(
            "GOOGLE_API_KEY",
            ""
        )

    try:

        secret = st.secrets.get(
            "GOOGLE_API_KEY",
            ""
        )

    except:

        secret = ""

    return (
        key
        or secret
        or os.getenv("GOOGLE_API_KEY", "")
    ).strip()


# ==========================================================
# JSON PARSER
# ==========================================================

def extract_json(text):

    text = text.strip()

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

    start = text.find("{")

    end = text.rfind("}")

    if start != -1:

        text = text[start:end+1]

    try:

        return json.loads(text)

    except:

        return {

            "dish_name":"Unknown",

            "confidence":"Low",

            "ingredients":[],

            "nutrition":{

                "calories":"N/A",

                "protein":"N/A",

                "fat":"N/A",

                "carbohydrates":"N/A"

            },

            "recipe":[],

            "alternatives":[],

            "portion_assumption":"Unknown"

        }


# ==========================================================
# GEMINI VISION
# ==========================================================

def generate_ai_response(
        image_bytes,
        user_context=""
):

    api_key = get_api_key()

    if api_key == "":

        raise RuntimeError(
            "Gemini API key missing."
        )

    client = genai.Client(
        api_key=api_key
    )
    prompt = f"""
You are an expert Indian food recognition and nutrition assistant.

The uploaded image is the PRIMARY source of truth.

An EfficientNetB0 deep learning classifier has already predicted the possible food categories.

The CNN predictions are only supporting information.

If the image and CNN prediction disagree,
ALWAYS trust the image.

User Information:
{user_context}

Return ONLY valid JSON in this format:

{{
"dish_name":"",
"confidence":"High/Medium/Low",

"ingredients":[
"",
""
],

"nutrition":{{
"calories":"",
"protein":"",
"fat":"",
"carbohydrates":""
}},

"recipe":[
"",
"",
""
],

"alternatives":[
"",
"",
""
],

"portion_assumption":""
}}

Rules:

1. Identify the food from the IMAGE.

2. Ignore wrong CNN predictions.

3. Nutrition values should be realistic.

4. Recipe should contain 5–7 short steps.

5. Alternatives should be healthier foods whenever possible.

6. Output ONLY JSON.
"""

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=[

            prompt,

            types.Part.from_bytes(

                data=image_bytes,

                mime_type="image/jpeg"

            )

        ]

    )

    if response is None:

        raise RuntimeError(
            "Gemini returned nothing."
        )

    if response.text is None:

        raise RuntimeError(
            "Gemini returned empty response."
        )

    return extract_json(
        response.text
    )


# ==========================================================
# NUTRITION PARSER
# ==========================================================

def nutrition_number(value):

    if isinstance(
        value,
        (int,float)
    ):

        return float(value)

    value = str(value)

    digits = ""

    for ch in value:

        if ch.isdigit() or ch==".":

            digits += ch

        else:

            digits += " "

    for token in digits.split():

        try:

            return float(token)

        except:

            pass

    return 0.0
# ==========================================================
# BMI RECOMMENDATION ENGINE
# ==========================================================

def generate_recommendation(
        bmi_category,
        food_analysis
):

    nutrition = food_analysis.get(
        "nutrition",
        {}
    )

    calories = nutrition_number(
        nutrition.get(
            "calories",
            0
        )
    )

    protein = nutrition_number(
        nutrition.get(
            "protein",
            0
        )
    )

    fat = nutrition_number(
        nutrition.get(
            "fat",
            0
        )
    )

    carbs = nutrition_number(
        nutrition.get(
            "carbohydrates",
            0
        )
    )

    score = 0

    reasons = []

    if calories > 700:

        score += 2

        reasons.append(
            "Very high calorie food"
        )

    elif calories > 450:

        score += 1

        reasons.append(
            "Moderately high calories"
        )

    if fat > 30:

        score += 1

        reasons.append(
            "High fat content"
        )

    if carbs > 80:

        score += 1

        reasons.append(
            "High carbohydrate content"
        )

    if protein > 20:

        score -= 1

        reasons.append(
            "Good protein source"
        )

    if bmi_category == "Obese":

        score += 2

        reasons.append(
            "BMI indicates obesity"
        )

    elif bmi_category == "Overweight":

        score += 1

        reasons.append(
            "BMI indicates overweight"
        )

    elif bmi_category == "Underweight":

        score -= 1

    if score <= 0:

        verdict = "Safe to Eat"

    elif score <= 2:

        verdict = "Eat in Moderation"

    else:

        verdict = "Avoid Frequently"

    suggestions = [

        "Reduce oil and butter.",

        "Prefer grilled or steamed preparation.",

        "Control portion size.",

        "Add vegetables or salad.",

        "Avoid sugary drinks with this meal."

    ]

    if bmi_category == "Underweight":

        suggestions.append(
            "Increase healthy protein intake."
        )

    if bmi_category in [

        "Overweight",

        "Obese"

    ]:

        suggestions.append(
            "Increase daily physical activity."
        )

    return {

        "verdict": verdict,

        "reason": ", ".join(reasons),

        "suggestions": suggestions

    }


# ==========================================================
# STREAMLIT HELPERS
# ==========================================================

def render_card(title, value):

    st.markdown(
        f"### {title}"
    )

    st.info(value)


def render_list(items):

    if isinstance(items, list):

        for item in items:

            st.markdown(
                f"- {item}"
            )

    else:

        st.write(items)
# ==========================================================
# MAIN APP
# ==========================================================

def main():

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🍽️",
        layout="wide"
    )

    st.title(APP_TITLE)

    st.markdown(
        "### EfficientNetB0 + Gemini AI Food Recognition"
    )

    # ==============================
    # Sidebar
    # ==============================

    with st.sidebar:

        st.header("User Details")

        age = st.number_input(
            "Age",
            1,
            100,
            25
        )

        gender = st.selectbox(

            "Gender",

            ["Male","Female","Other"]

        )

        height = st.number_input(

            "Height (m)",

            0.5,

            2.5,

            1.70,

            step=0.01

        )

        weight = st.number_input(

            "Weight (kg)",

            10,

            250,

            70

        )

        bmi = calculate_bmi(

            weight,

            height

        )

        bmi_category = classify_bmi(

            bmi

        )

        st.metric(

            "BMI",

            f"{bmi:.2f}",

            bmi_category

        )

    # ==============================

    col1,col2 = st.columns([1,1.2])

    with col1:

        option = st.radio(

            "Select Image Source",

            [

                "Upload Image",

                "Camera"

            ]

        )

        if option=="Upload Image":

            uploaded_file = st.file_uploader(

                "Choose Food Image",

                type=[

                    "jpg",

                    "jpeg",

                    "png"

                ]

            )

        else:

            uploaded_file = st.camera_input(

                "Capture Food"

            )

        notes = st.text_area(

            "Extra Notes (Optional)",

            ""

        )

    image = None

    image_bytes = None

    predictions = None

    food_name = ""

    confidence = 0

    if uploaded_file is not None:

        try:

            image,image_bytes = preprocess_image(

                uploaded_file

            )

            predictions = predict_food_category(

                image

            )

            food_name = predictions[0]["food"]

            confidence = predictions[0]["confidence"]

            with col1:

                st.image(

                    image,

                    use_container_width=True
                )

            st.success("Prediction Complete")

        except Exception as e:

            st.error(e)
    with col2:

        st.subheader("Food Analysis")

        if st.button(
            "🔍 Analyze Food",
            use_container_width=True
        ):

            if image_bytes is None:

                st.warning(
                    "Please upload an image first."
                )

            else:

                cnn_context = f"""

Top 5 EfficientNet Predictions

"""

                for i, pred in enumerate(predictions, start=1):

                    cnn_context += (
                        f"{i}. "
                        f"{pred['food']} "
                        f"({pred['confidence']:.2f}%)\n"
                    )

                profile = f"""

Age : {age}

Gender : {gender}

Height : {height}

Weight : {weight}

BMI : {bmi:.2f}

BMI Category : {bmi_category}

Notes : {notes}

{cnn_context}

"""

                with st.spinner(

                    "Analyzing using Gemini..."

                ):

                    try:

                        analysis = generate_ai_response(

                            image_bytes,

                            profile

                        )

                    except Exception as e:

                        st.error(e)

                        st.stop()

                recommendation = generate_recommendation(

                    bmi_category,

                    analysis

                )

                # ===========================
                # CNN Prediction
                # ===========================

                st.success("CNN Prediction")

                st.write(

                    f"### {food_name}"

                )

                st.write(

                    f"Confidence : "

                    f"{confidence:.2f}%"

                )

                st.divider()

                # ===========================
                # Gemini
                # ===========================

                st.success("Gemini Analysis")

                st.write(

                    "###",

                    analysis.get(

                        "dish_name",

                        "Unknown"

                    )

                )

                st.write(

                    "Confidence :",

                    analysis.get(

                        "confidence",

                        "N/A"

                    )

                )

                nutrition = analysis.get(

                    "nutrition",

                    {}

                )

                c1,c2,c3,c4 = st.columns(4)

                with c1:

                    render_card(

                        "Calories",

                        nutrition.get(

                            "calories",

                            "N/A"

                        )

                    )

                with c2:

                    render_card(

                        "Protein",

                        nutrition.get(

                            "protein",

                            "N/A"

                        )

                    )

                with c3:

                    render_card(

                        "Fat",

                        nutrition.get(

                            "fat",

                            "N/A"

                        )

                    )

                with c4:

                    render_card(

                        "Carbohydrates",

                        nutrition.get(

                            "carbohydrates",

                            "N/A"

                        )

                    )

                st.divider()

                tab1,tab2,tab3,tab4 = st.tabs(

                    [

                        "❤️ Recommendation",

                        "🥗 Ingredients",

                        "👨‍🍳 Recipe",

                        "🔄 Alternatives"

                    ]

                )

                with tab1:

                    st.subheader(

                        recommendation["verdict"]

                    )

                    st.write(

                        recommendation["reason"]

                    )

                    st.markdown(

                        "### Suggestions"

                    )

                    render_list(

                        recommendation["suggestions"]

                    )

                with tab2:

                    render_list(

                        analysis.get(

                            "ingredients",

                            []

                        )

                    )

                with tab3:

                    render_list(

                        analysis.get(

                            "recipe",

                            []

                        )

                    )

                with tab4:

                    render_list(

                        analysis.get(

                            "alternatives",

                            []

                        )

                    )

    st.divider()

    st.caption(

        "⚠️ Nutrition values are AI estimates only and should not replace professional medical advice."

    )


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    main()