# ==========================================================
# NutriVision AI v2.0
# ==========================================================

import streamlit as st
from utils import *
from components import *

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Vision-to-Nutrition",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CSS
# ==========================================================

with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )


# ==========================================================
# SESSION STATE
# ==========================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "top5" not in st.session_state:
    st.session_state.top5 = None

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/480/artificial-intelligence.png",
        width=90
    )

    st.title("Vision-to-Nutrition")
    st.caption("Intelligent Food Analysis using EfficientNet & Gemini")

    st.divider()

    st.markdown("### 🤖 AI Model")

    st.success("EfficientNetB0")

    st.info("80 Indian Food Classes")

    st.info("Gemini 2.5 Flash")

    st.info("Validation Accuracy\n\n69.92%")

    st.divider()

    st.markdown("### 👨‍💻 Developer")

    st.write("Yashwanth")

    st.divider()


# ==========================================================
# HERO
# ==========================================================

hero_header()

# ==========================================================
# TOP CONTROL PANEL
# ==========================================================

st.markdown("## 📤 Upload & Analyze")

c1,c2,c3,c4,c5,c6,c7 = st.columns(
    [2.4,2.2,1,1,1,1,1.5]
)

# Upload

with c1:

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )

# Camera

with c2:

    camera_file = st.camera_input(
        "Camera"
    )

if uploaded_file is None and camera_file is not None:
    uploaded_file = camera_file

# User Details

with c3:

    age = st.number_input(
        "Age",
        10,
        100,
        22
    )

with c4:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

with c5:

    height_cm = st.number_input(
        "Height",
        120,
        220,
        170
    )

with c6:

    weight = st.number_input(
        "Weight",
        30,
        180,
        70
    )

height = height_cm / 100

bmi = calculate_bmi(
    weight,
    height
)

bmi_status = bmi_category(
    bmi
)

# Analyze Button

with c7:

    st.write("")

    analyze_btn = st.button(
        "🚀 Analyze",
        use_container_width=True,
        disabled=uploaded_file is None
    )

st.write("")

# ==========================================================
# MAIN DASHBOARD
# ==========================================================

left_col, right_col = st.columns(
    [1.2,1]
)
# ==========================================================
# LEFT PANEL
# ==========================================================

with left_col:

    st.markdown("## 📷 Food Image")

    if uploaded_file is None:

        st.markdown("""
        <div class="glass-card"
        style="
        height:450px;
        display:flex;
        justify-content:center;
        align-items:center;
        flex-direction:column;
        ">

        <h1 style="font-size:80px;">🍽️</h1>

        <h3>No Image Selected</h3>

        <p>
        Upload or capture a food image to begin.
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    else:

        image, image_bytes = preprocess_image(uploaded_file)

        st.image(
            image,
            use_container_width=True
        )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right_col:

    st.markdown("## 🤖 AI Prediction")

    if uploaded_file is None:

        st.markdown("""
        <div class="glass-card"
        style="
        height:450px;
        display:flex;
        justify-content:center;
        align-items:center;
        flex-direction:column;
        ">

        <h1 style="font-size:80px;">🧠</h1>

        <h3>Waiting for Image...</h3>

        <p>
        Prediction will appear here.
        </p>

        </div>
        """,
        unsafe_allow_html=True)

    else:

        if analyze_btn:

            with st.spinner("Analyzing Food..."):

                prediction, confidence, top5 = predict_food(image)

                st.session_state.prediction = prediction

                st.session_state.confidence = confidence

                st.session_state.top5 = top5

                user_context = f"""

Age : {age}

Gender : {gender}

Height : {height_cm}

Weight : {weight}

BMI : {bmi:.2f}

BMI Status : {bmi_status}

Predicted Food : {prediction}

Confidence : {confidence:.2f}

"""

                analysis = generate_ai_response(
                    image_bytes,
                    user_context
                )

                st.session_state.analysis = analysis

# ==========================================================
# SHOW PREDICTION
# ==========================================================

if st.session_state.prediction is not None:

    st.write("")

    dish_name = st.session_state.analysis.get(
    "dish_name",
    st.session_state.prediction.replace("_", " ").title()
)

    st.markdown(f"""
    <div class="prediction-card">

    <h4>🍽 Identified Dish</h4>

    <h1>{dish_name}</h1>

    <p>
    Identified using Gemini AI after visual verification.
    </p>

    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card">

    <h4>🧠 CNN Prediction</h4>

    <p><b>{st.session_state.prediction.replace("_"," ").title()}</b></p>

    <p>
    Confidence: <b>{st.session_state.confidence:.2f}%</b>
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    top_predictions(

        st.session_state.top5

    )

# ==========================================================
# NUTRITION CARDS
# ==========================================================

analysis = st.session_state.analysis

if analysis:

    st.write("")

    section_title("🍽 Nutrition Information")

    nutrition = analysis.get("nutrition",{})

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        metric_card(
            "🔥",
            "Calories",
            nutrition.get("calories","--")
        )

    with c2:

        metric_card(
            "🥩",
            "Protein",
            nutrition.get("protein","--")
        )

    with c3:

        metric_card(
            "🧈",
            "Fat",
            nutrition.get("fat","--")
        )

    with c4:

        metric_card(
            "🌾",
            "Carbs",
            nutrition.get("carbohydrates","--")
        )
# ==========================================================
# AI INSIGHTS
# ==========================================================

if analysis:

    st.write("")

    section_title("🤖 AI Insights")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🥗 Ingredients",
            "👨‍🍳 Recipe",
            "❤️ Recommendation",
            "🔄 Alternatives"
        ]
    )

    # ======================================================
    # INGREDIENTS
    # ======================================================

    with tab1:

        ingredients = analysis.get(
            "ingredients",
            []
        )

        info_card(
            "Ingredients",
            ingredients
        )

    # ======================================================
    # RECIPE
    # ======================================================

    with tab2:

        recipe = analysis.get(
            "recipe",
            []
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.subheader("👨‍🍳 Recipe")

        if recipe:

            for i, step in enumerate(recipe, 1):

                st.markdown(
                    f"""
### Step {i}

{step}

---
"""
                )

        else:

            st.info(
                "Recipe unavailable."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # ======================================================
    # RECOMMENDATION
    # ======================================================

    with tab3:

        recommendation = generate_recommendation(
            bmi_status,
            analysis
        )

        recommendation_card(
            recommendation
        )

    # ======================================================
    # ALTERNATIVES
    # ======================================================

    with tab4:

        alternatives = analysis.get(
            "alternatives",
            []
        )

        info_card(
            "Healthier Alternatives",
            alternatives
        )

# ==========================================================
# FOOTER
# ==========================================================

st.write("")
st.divider()

st.markdown(
    """
    <center>

    <h3>🍽️ Vision-to-Nutrition</h3>
    An EfficientNet–Gemini Framework for Intelligent Food Analysis
    Powered by EfficientNetB0 + Gemini AI

    </center>
    """,
    unsafe_allow_html=True
)
