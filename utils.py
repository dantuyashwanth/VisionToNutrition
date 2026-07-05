# ==========================================================
# utils.py
# NutriVision AI Backend
# ==========================================================

import io
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from dotenv import load_dotenv, dotenv_values

from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications.efficientnet import preprocess_input

from google import genai
from google.genai import types

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_efficientnet.keras"

CLASS_PATH = BASE_DIR / "class_names.json"

GEMINI_MODEL = "gemini-2.5-flash"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_food_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


food_model = load_food_model()

# ==========================================================
# LOAD CLASSES
# ==========================================================

@st.cache_data
def load_classes():

    with open(CLASS_PATH, "r") as f:

        classes = json.load(f)

    return classes


CLASS_NAMES = load_classes()

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(uploaded_file):

    if uploaded_file is None:

        raise ValueError("Please upload an image.")

    try:

        img = Image.open(uploaded_file)

        img.verify()

        uploaded_file.seek(0)

        img = Image.open(uploaded_file).convert("RGB")

    except (UnidentifiedImageError, OSError):

        raise ValueError("Invalid image.")

    preview = img.copy()

    preview.thumbnail((1024,1024))

    buffer = io.BytesIO()

    preview.save(

        buffer,

        format="JPEG",

        quality=90

    )

    return preview, buffer.getvalue()

# ==========================================================
# CNN PREPROCESS
# ==========================================================

def prepare_for_model(img):

    img = img.resize((224,224))

    img = np.array(img).astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img,axis=0)

    return img

# ==========================================================
# FOOD PREDICTION
# ==========================================================

def predict_food(image):

    x = prepare_for_model(image)

    preds = food_model.predict(

        x,

        verbose=0

    )[0]

    top5 = preds.argsort()[-5:][::-1]

    prediction = CLASS_NAMES[top5[0]]

    confidence = float(preds[top5[0]]*100)

    results=[]

    for idx in top5:

        results.append({

            "food":CLASS_NAMES[idx],

            "confidence":round(

                float(preds[idx]*100),

                2

            )

        })

    return prediction,confidence,results

# ==========================================================
# BMI
# ==========================================================

def calculate_bmi(weight,height):

    return weight/(height**2)


def bmi_category(bmi):

    if bmi<18.5:

        return "Underweight"

    elif bmi<25:

        return "Healthy"

    elif bmi<30:

        return "Overweight"

    else:

        return "Obese"

# ==========================================================
# API KEY
# ==========================================================

def get_api_key():

    env_file = BASE_DIR / ".env"

    load_dotenv(env_file)

    key=""

    if env_file.exists():

        key = dotenv_values(

            env_file

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

        secret=""

    return (

        key

        or secret

        or os.getenv(

            "GOOGLE_API_KEY",

            ""

        )

    ).strip()
# ==========================================================
# JSON PARSER
# ==========================================================

def extract_json(text):

    text = text.strip()

    if text.startswith("```"):

        text = text.replace("```json", "")
        text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

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
# GEMINI CLIENT
# ==========================================================

def get_gemini_client():

    api_key = get_api_key()

    if api_key == "":

        raise RuntimeError(
            "Gemini API Key Missing."
        )

    return genai.Client(api_key=api_key)

# ==========================================================
# GEMINI ANALYSIS
# ==========================================================

def generate_ai_response(

        image_bytes,

        user_context=""

):

    client = get_gemini_client()

    prompt = f"""

You are an expert Indian nutritionist.

The uploaded image is the PRIMARY source.

EfficientNet predictions are only supporting information.

User Information:

{user_context}

Return ONLY valid JSON.

{{
"dish_name":"",
"confidence":"",

"ingredients":[
"",
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

Do NOT return markdown.

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
            "Empty Gemini response."
        )

    return extract_json(response.text)

# ==========================================================
# NUTRITION VALUE PARSER
# ==========================================================

def nutrition_number(value):

    if isinstance(value,(int,float)):

        return float(value)

    value = str(value)

    digits = ""

    for ch in value:

        if ch.isdigit() or ch == ".":

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
# RECOMMENDATION ENGINE
# ==========================================================

def generate_recommendation(

        bmi_status,

        analysis

):

    nutrition = analysis.get(

        "nutrition",

        {}

    )

    calories = nutrition_number(

        nutrition.get("calories",0)

    )

    protein = nutrition_number(

        nutrition.get("protein",0)

    )

    fat = nutrition_number(

        nutrition.get("fat",0)

    )

    carbs = nutrition_number(

        nutrition.get("carbohydrates",0)

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

            "High carbohydrates"

        )

    if protein > 20:

        score -= 1

        reasons.append(

            "Good protein source"

        )

    if bmi_status == "Obese":

        score += 2

        reasons.append(

            "BMI indicates obesity"

        )

    elif bmi_status == "Overweight":

        score += 1

        reasons.append(

            "BMI indicates overweight"

        )

    elif bmi_status == "Underweight":

        score -= 1

    if score <= 0:

        verdict = "Safe to Eat"

    elif score <= 2:

        verdict = "Eat in Moderation"

    else:

        verdict = "Avoid Frequently"

    suggestions = [

        "Reduce oil and butter.",

        "Control portion size.",

        "Add vegetables or salad.",

        "Avoid sugary drinks.",

        "Drink more water."

    ]

    if bmi_status == "Underweight":

        suggestions.append(

            "Increase healthy protein intake."

        )

    if bmi_status in [

        "Overweight",

        "Obese"

    ]:

        suggestions.append(

            "Walk at least 30 minutes daily."

        )

    return {

        "verdict": verdict,

        "reason": ", ".join(reasons),

        "suggestions": suggestions

    }

# ==========================================================
# STREAMLIT HELPERS
# ==========================================================

def render_list(items):

    if isinstance(items,list):

        for item in items:

            st.markdown(f"- {item}")

    else:

        st.write(items)


def render_metric(title,value):

    st.metric(

        label=title,

        value=value

    )
