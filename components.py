# ==========================================================
# components.py
# NutriVision AI UI Components
# ==========================================================

import streamlit as st

# ==========================================================
# HERO HEADER
# ==========================================================

def hero_header():

    st.markdown(
        """
        <div class="hero-card">
            <h1>🍽️ Vision-to-Nutrition</h1>
            <h3>An EfficientNet–Gemini Framework</h3>
            <p>
                Intelligent Food Analysis • Nutrition Insights •
                Health Recommendations
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# SECTION TITLE
# ==========================================================

def section_title(title):

    st.markdown(f"""
    <h2 style="
    margin-top:25px;
    color:white;
    ">
    {title}
    </h2>
    """,
    unsafe_allow_html=True)


# ==========================================================
# IMAGE CARD
# ==========================================================

def image_card(image):

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ==========================================================
# PREDICTION CARD
# ==========================================================

def prediction_card(food, confidence):

    st.markdown(f"""
    <div class="prediction-card">

    <h4>
    🍽 Prediction
    </h4>

    <h1>
    {food.replace("_"," ").title()}
    </h1>

    <h2 style="color:#00E5A8;">
    {confidence:.2f}%
    </h2>

    </div>
    """,
    unsafe_allow_html=True)


# ==========================================================
# TOP 5 PREDICTIONS
# ==========================================================

def top_predictions(results):

    st.subheader("📈 Top Predictions")

    for item in results:

        st.markdown(f"""
        <div style="
        display:flex;
        justify-content:space-between;
        margin-top:12px;
        ">

        <span>

        {item['food'].replace("_"," ").title()}

        </span>

        <b>

        {item['confidence']:.2f}%

        </b>

        </div>
        """,
        unsafe_allow_html=True)

        st.progress(item["confidence"]/100)


# ==========================================================
# METRIC CARD
# ==========================================================

def metric_card(icon,title,value):

    st.markdown(f"""
    <div class="metric-card">

    <h1>{icon}</h1>

    <h2>{value}</h2>

    <p>{title}</p>

    </div>
    """,
    unsafe_allow_html=True)


# ==========================================================
# BMI CARD
# ==========================================================

def bmi_card(bmi,status):

    if status=="Healthy":

        color="#16A34A"

    elif status=="Underweight":

        color="#3B82F6"

    elif status=="Overweight":

        color="#F59E0B"

    else:

        color="#EF4444"

    st.markdown(f"""
    <div class="glass-card">

    <h3>
    ❤️ BMI
    </h3>

    <h1 style="color:{color};">
    {bmi:.1f}
    </h1>

    <h4>
    {status}
    </h4>

    </div>
    """,
    unsafe_allow_html=True)


# ==========================================================
# INFO CARD
# ==========================================================

def info_card(title,items):

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    st.subheader(title)

    if isinstance(items,list):

        for item in items:

            st.markdown(f"✔ {item}")

    else:

        st.write(items)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ==========================================================
# RECOMMENDATION CARD
# ==========================================================

def recommendation_card(result):

    verdict = result["verdict"]

    color = "#16A34A"

    if verdict=="Eat in Moderation":

        color="#F59E0B"

    elif verdict=="Avoid Frequently":

        color="#EF4444"

    st.markdown(f"""
    <div class="recommend-card">

    <h2 style="color:{color};">

    {verdict}

    </h2>

    <p>

    {result['reason']}

    </p>

    </div>
    """,
    unsafe_allow_html=True)

    st.markdown("### Suggestions")

    for tip in result["suggestions"]:

        st.success(tip)


# ==========================================================
# FOOTER
# ==========================================================

def footer():

    st.markdown("""

    <br><br>

    <hr>

    <center>

    NutriVision AI v2.0

    <br>

    Powered by EfficientNetB0 + Gemini AI

    </center>

    """,
    unsafe_allow_html=True)