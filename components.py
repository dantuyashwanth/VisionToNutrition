# ==========================================================
# components.py
# NutriVision AI UI Reusable Components
# ==========================================================

import streamlit as st

def hero_header():
    st.markdown("""
        <div class="hero-card">
            <h1>Vision To Nutrition</h1>
            <h3>Futuristic Food Analysis & Metabolic Tracking</h3>
            <p>Powered by Deep CNN EfficientNetB0 architecture combined with advanced Gemini 2.5 Flash Vision Models.</p>
        </div>
    """, unsafe_allow_html=True)

def image_card(image_ready: bool = False):
    status_badge = '<div style="background:#10B981;color:white;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">Image Ready</div>' if image_ready else '<div style="background:#EF4444;color:white;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">Waiting...</div>'
    st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                <span style="color:#94A3B8; font-weight:600; font-size:14px;">INPUT STREAM</span>
                {status_badge}
            </div>
        </div>
    """, unsafe_allow_html=True)

def prediction_card(food_name: str, confidence: float):
    st.markdown(f"""
        <div class="prediction-card">
            <span class="top-badge">Top Prediction</span>
            <h4 style="margin:0; color:#94A3B8;">CNN Classification</h4>
            <h1 style="margin:5px 0 15px 0;">{food_name.replace('_', ' ').title()}</h1>
            <div style="display:flex; justify-content:space-between; font-size:14px; color:#94A3B8;">
                <span>Confidence Score</span>
                <span style="color:#10B981; font-weight:700;">{confidence:.2f}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {confidence}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def top_predictions(top5: list):
    st.markdown('<div class="glass-card"><h4>📊 Top Model Hypotheses</h4>', unsafe_allow_html=True)
    for i, item in enumerate(top5, start=1):
        st.markdown(f"""
            <div class="top5-item">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div class="rank-circle">{i}</div>
                    <span style="font-weight:500; color:#E2E8F0;">{item['food'].replace('_', ' ').title()}</span>
                </div>
                <span style="color:#8B5CF6; font-weight:600; font-size:14px;">{item['confidence']:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def metric_card(icon: str, title: str, value: str):
    css_class = "metric-card"
    if "Calorie" in title: css_class += " calorie"
    elif "Protein" in title: css_class += " protein"
    elif "Fat" in title: css_class += " fat"
    elif "Carb" in title: css_class += " carbs"
    
    # Appends standard baseline unit underneath for visual clarity
    unit_label = "Per 100g Serve"
    
    st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:28px;">{icon}</div>
            <h3>{title}</h3>
            <h1 style="margin: 5px 0 0 0 !important; font-size: 26px !important;">{value}</h1>
            <span style="color:#64748B; font-size:12px; font-weight:500; display:block; margin-top:4px;">{unit_label}</span>
        </div>
    """, unsafe_allow_html=True)

def info_card(title: str, items: list, icon: str = "📋"):
    st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:white; margin-bottom:20px; display:flex; align-items:center; gap:10px;">
                <span>{icon}</span> {title}
            </h3>
    """, unsafe_allow_html=True)
    for item in items:
        st.markdown(f"""
            <div class="list-item">
                <div class="bullet-icon">✦</div>
                <div class="list-text">{item}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def health_score_card(score: float, verdict: str, description: str):
    percentage = min(max(score * 10, 0), 100)
    st.markdown(f"""
        <div class="glass-card">
            <h3>📈 Dynamic Health Assessment</h3>
            <div class="score-box">
                <div class="circle-meter" style="--pg: {percentage}deg;">
                    <div class="circle-text">
                        {score:.1f}
                        <span class="circle-sub">Scale of 10</span>
                    </div>
                </div>
                <div style="flex: 1; min-width: 250px;">
                    <h2 style="margin:0 0 8px 0; color:#10B981; font-weight:700;">{verdict}</h2>
                    <p style="color:#94A3B8; font-size:14px; margin:0; line-height:1.6;">{description}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def bmi_card(bmi: float, category: str):
    color_map = {"Underweight": "#3B82F6", "Healthy": "#10B981", "Overweight": "#F59E0B", "Obese": "#EF4444"}
    color = color_map.get(category, "#8B5CF6")
    st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding: 20px !important;">
            <span style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">Computed Metabolic Profile</span>
            <h1 style="font-size:44px !important; margin:10px 0 0 0 !important; color:{color};">{bmi:.1f}</h1>
            <p style="color:#E2E8F0; font-size:15px; font-weight:600; margin:5px 0 0 0;">{category}</p>
        </div>
    """, unsafe_allow_html=True)

def food_not_detected_card(obj_name: str, desc: str, reason: str):
    st.markdown(f"""
        <div class="glass-card" style="border: 1px solid rgba(239, 68, 68, 0.3) !important;">
            <div style="background:rgba(239,68,68,0.1); color:#EF4444; padding:10px 16px; border-radius:12px; font-weight:700; display:inline-block; margin-bottom:15px;">
                ⚠️ Food Object Integrity Verification Failed
            </div>
            <h2 style="margin:0 0 10px 0; color:white;">Detected Object: <span style="color:#EF4444;">{obj_name}</span></h2>
            <p style="color:#94A3B8; line-height:1.6; font-size:15px; margin-bottom:15px;"><b>Visual Signature:</b> {desc}</p>
            <p style="color:#64748B; line-height:1.6; font-size:14px; margin:0; background:rgba(0,0,0,0.2); padding:12px; border-radius:8px;">
                <b>Reasoning Engine Output:</b> {reason}
            </p>
        </div>
    """, unsafe_allow_html=True)
