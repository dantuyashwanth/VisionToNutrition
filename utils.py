# ==========================================================
# utils.py
# NutriVision AI Analytical Core Pipeline
# ==========================================================

import io
import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
from tensorflow.keras.applications.efficientnet import preprocess_input
from dotenv import load_dotenv  # 1. Add this import

# 2. Invoke it immediately to read your local .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_efficientnet.keras"
CLASS_PATH = BASE_DIR / "class_names.json"

@st.cache_resource
def load_food_model():
    """Loads the physical pre-trained EfficientNetB0 model asset."""
    if MODEL_PATH.exists():
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading keras model: {e}")
            st.stop()
    else:
        st.error(f"Critical Error: '{MODEL_PATH.name}' not found in root directory!")
        st.stop()

@st.cache_data
def load_class_names():
    """Loads all 80 food category labels dynamically from class_names.json."""
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r") as f:
            return json.load(f)
    else:
        st.error(f"Critical Error: '{CLASS_PATH.name}' missing!")
        st.stop()

def get_api_key():
    try:
        # Check Streamlit secrets first, then standard environment variables
        key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        
        # Defensive check: grab lowercase/mixed case variations from the .env file if upper fails
        if not key:
            key = os.getenv("google_api_key", os.getenv("google_API_key", ""))
            
        return key.strip()
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "").strip()

def get_gemini_client():
    key = get_api_key()
    if not key:
        st.error("Missing GOOGLE_API_KEY configuration.")
        st.stop()
    return genai.Client(api_key=key)

def calculate_bmi(weight: float, height_cm: float) -> tuple[float, str]:
    height_m = height_cm / 100.0
    bmi = weight / (height_m ** 2)
    if bmi < 18.5: cat = "Underweight"
    elif bmi < 25.0: cat = "Healthy"
    elif bmi < 30.0: cat = "Overweight"
    else: cat = "Obese"
    return bmi, cat

def parse_json_safely(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[-1].split("```")[0]
    try:
        return json.loads(cleaned)
    except Exception:
        return {}
@st.cache_data(show_spinner=False)
def detect_food_image(image_bytes: bytes) -> dict:
    """Workflow Stage: Gemini Image Verification Strategy with Error Resilience"""
    client = get_gemini_client()
    prompt = """
    Analyze this image closely. Determine if the primary foreground item is edible food.
    Return response STRICTLY inside structural JSON matching this format:
    {
      "is_food": true or false,
      "object_name": "Name of the detected object",
      "description": "Natural clear description",
      "reason": "Clear analytical deduction logic details"
    }
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
        )
        return parse_json_safely(response.text)
    except Exception as e:
        # Check if it's a 503/server capacity issue and send a flag
        return {"api_error": True, "message": "Gemini servers are currently overloaded. Please wait a few seconds and try again."}

def predict_food(image_pil: Image.Image, classes: list) -> tuple[str, float, list]:
    """Workflow Stage: EfficientNetB0 CNN Food Classification"""
    model = load_food_model()
    # Preprocess image precisely for EfficientNetB0 expectation standard
    img_resized = image_pil.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    predictions = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:5]
    
    top_prediction = classes[top_indices[0]]
    top_confidence = float(predictions[top_indices[0]] * 100.0)
    
    predictions_list = []
    for idx in top_indices:
        predictions_list.append({
            "food": classes[idx],
            "confidence": float(predictions[idx] * 100.0)
        })
    return top_prediction, top_confidence, predictions_list
@st.cache_data(show_spinner=False)  # <-- Add this decorator
def generate_ai_response(image_bytes: bytes, cnn_prediction: str, confidence: float, bmi_status: str, age: int, gender: str) -> dict:
    """Workflow Stage: Gemini AI Food Analysis mapped to strict per-100g standardized units"""
    client = get_gemini_client()
    prompt = f"""
    You are an expert clinical nutritionist analyzing a food image. 
    Cross-examine the visual signatures against our local CNN model insights:
    - CNN Predicted Label: '{cnn_prediction}' ({confidence:.1f}% confidence convergence)
    
    User Profile Context:
    - Age: {age} | Gender: {gender} | Metabolic Status: {bmi_status}
    
    CRITICAL METRIC REQUIREMENT: Calculate ALL nutritional metrics strictly standardized PER 100g SERVING.
    Do NOT estimate the size of the total plate. Provide values relative to a 100g reference amount.
    
    Synthesize the information and respond STRICTLY in JSON mirroring this exact structure:
    {{
      "dish_name": "Verified/Corrected target food name",
      "ingredients": ["Ingredient breakdown normalized per 100g"],
      "nutrition": {{
         "calories": "e.g., 340 kcal per 100g",
         "protein": "e.g., 14g per 100g",
         "fat": "e.g., 9g per 100g",
         "carbohydrates": "e.g., 45g per 100g"
      }},
      "recipe": ["Sequential preparation steps"],
      "alternatives": ["Optimized healthier alternatives"],
      "health_score": 8.5,
      "verdict": "Eat in Moderation",
      "suggestions": ["Portion control adjustments matched to user profile context"]
    }}
    """
    try:
        # Inside utils.py, look for your model calls:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # <-- Update this parameter string in both API functions
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
        )
        return parse_json_safely(response.text)
    except Exception as e:
        return {"api_error": True}
