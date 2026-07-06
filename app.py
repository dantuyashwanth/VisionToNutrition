import streamlit as st
from PIL import Image
import io

# Critical architectural imports
from utils import load_class_names, calculate_bmi, detect_food_image, generate_ai_response, predict_food
import components

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Vision-to-Nutrition",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded" # Enforces default deployment expansion
)

# 2. LOAD CUSTOM UI THEME LAYOUT
try:
    with open("style.css", "r") as css_data:
        st.markdown(f"<style>{css_data.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# 3. INITIALIZE ALL ENGINE STATE VARIABLES (Fixes the AttributeError)
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "top5" not in st.session_state:
    st.session_state.top5 = None
if "verification_audit" not in st.session_state:
    st.session_state.verification_audit = None

# App Master Header Layout Component
components.hero_header()

# System Engine State Handlers Loading Preconditions
label_indices = load_class_names()

# 4. PREMIUM SIDEBAR SETUP BLOCK FRAME ARCHITECTURE
with st.sidebar:
    st.markdown('<h2 style="margin-top:0;">🍽️ System Control Workspace</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🧬 User Profile Metadata")
    input_age = st.number_input("Age Profile (Years)", min_value=1, max_value=120, value=25)
    input_gender = st.selectbox("Biological Sex", ["Male", "Female", "Non-Binary"])
    input_height = st.number_input("Height Metric (cm)", min_value=50.0, max_value=250.0, value=170.0)
    input_weight = st.number_input("Weight Metric (kg)", min_value=10.0, max_value=300.0, value=75.0)
    
    # Live BMI Calculation Module Evaluation Processing
    user_bmi, user_category = calculate_bmi(input_weight, input_height)
    st.markdown("<br>", unsafe_allow_html=True)
    components.bmi_card(user_bmi, user_category)
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Matrix Core Status")
    st.markdown("""
        <div style="font-size:13px; color:#94A3B8; line-height:1.8;">
            🟢 EfficientNetB0 Architecture: <span style="color:#10B981;font-weight:600;">READY</span><br>
            🟢 Gemini 2.5 Flash Interface: <span style="color:#10B981;font-weight:600;">CONNECTED</span><br>
            🟢 Core Dashboard Framework: <span style="color:#10B981;font-weight:600;">ONLINE</span>
        </div>
    """, unsafe_allow_html=True)

# 5. PRESENTATION GRID LAYOUT ARCHITECTURE (Side-by-Side Segment)
left_panel, right_panel = st.columns([1.1, 1.3], gap="large")

with left_panel:
    st.markdown("### 📥 Image Capture Pipeline Input")
    source_selection = st.radio("Primary Sensor Selection Channel", ["Upload Image Signature Asset File", "Access System Device Camera Feed Layer"], label_visibility="collapsed")
    
    file_stream = None
    if "Upload Image" in source_selection:
        file_stream = st.file_uploader("Drop image target configuration variables stream", type=["jpg", "png", "jpeg"])
    else:
        file_stream = st.camera_input("Acquire active spatial vector capture stream frame context")
        
    if file_stream is not None:
        raw_image_data = file_stream.read()
        image_object = Image.open(io.BytesIO(raw_image_data)).convert("RGB")
        st.image(image_object, use_container_width=True, caption="Active Memory Frame Object Target Cache")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Core Pipeline Processing Engine Trigger Node
        compute_matrix = st.button("⚡ EXECUTE ALL LAYER MULTIMODAL INFERENCE PIPELINE")
    else:
        components.image_card(image_ready=False)

with right_panel:
    st.markdown("### 🖥️ Deep Interactive Evaluation Terminal")
    
    if file_stream is not None and 'compute_matrix' in locals() and compute_matrix:
        with st.spinner("Executing Pipeline Sequence: Ingesting -> Verifying..."):
            
            # Step 1: Gemini Image Verification Strategy Layer
            verification_audit = detect_food_image(raw_image_data)
            st.session_state.verification_audit = verification_audit
            
            if verification_audit.get("api_error"):
                st.error("☁️ Gemini API Capacity Alert")
                st.warning("The AI service is experiencing high traffic spikes. Please wait 5 seconds and click the Analyze button again.")
                st.session_state.analysis = None
            
            elif not verification_audit.get("is_food", True):
                components.food_not_detected_card(
                    verification_audit.get("object_name", "Unknown Entity"),
                    verification_audit.get("description", "No explicit sensory frame details generated."),
                    verification_audit.get("reason", "Structural vector anomaly evaluation failed signature metrics verification profiles pattern mismatch logic.")
                )
                st.session_state.analysis = verification_audit
            else:
                # Step 2: Spatial CNN Classification Mapping execution
                top_prediction, top_confidence, distribution_matrix = predict_food(image_object, label_indices)
                st.session_state.prediction = top_prediction
                st.session_state.confidence = top_confidence
                st.session_state.top5 = distribution_matrix
                
                # Step 3: Run Gemini AI Core Food Analysis combined with recommendation layers metrics engine
                payload_matrix = generate_ai_response(
                    image_bytes=raw_image_data,
                    cnn_prediction=top_prediction,
                    confidence=top_confidence,
                    bmi_status=user_category,
                    age=input_age,
                    gender=input_gender
                )
                st.session_state.analysis = payload_matrix

    # Persistent top-half rendering context for Classification alerts inside right panel
    if st.session_state.analysis and st.session_state.analysis.get("is_food", True) and not st.session_state.analysis.get("api_error"):
        true_dish_name = st.session_state.analysis.get("dish_name", st.session_state.prediction.replace('_', ' ').title())
        
        if st.session_state.confidence < 62.0:
            st.markdown(f"""
                <div class="glass-card" style="border: 1px solid rgba(245, 158, 11, 0.4) !important; background: rgba(245, 158, 11, 0.05) !important;">
                    <span style="background:#F59E0B; color:white; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; text-transform:uppercase;">⚠️ Multimodal Realignment Active</span>
                    <h4 style="margin:10px 0 2px 0; color:#94A3B8;">Verified Meal Identity</h4>
                    <h1 style="margin:0 0 10px 0; color:white; font-size:36px;">{true_dish_name}</h1>
                    <p style="color:#CBD5E1; font-size:14px; margin:0; line-height:1.5;">
                        <b>System Notice:</b> The local edge CNN predicted <i>{st.session_state.prediction.replace('_', ' ').title()}</i> with low confidence ({st.session_state.confidence:.1f}%). 
                        The cloud-level multimodal vision alignment layer has intercepted and corrected the output to ensure data integrity.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            components.prediction_card(st.session_state.prediction, st.session_state.confidence)
            
        components.top_predictions(st.session_state.top5)
    elif file_stream is None:
        st.info("System Engine state idle. Provide media configuration payload target stream elements to start computational execution layers parsing threads routines.")

# ==========================================================
# 6. BREAK OUT OF COLUMNS - FULL WIDTH LOWER DASHBOARD SECTION
# ==========================================================
if st.session_state.analysis and st.session_state.analysis.get("is_food", True) and not st.session_state.analysis.get("api_error"):
    payload_matrix = st.session_state.analysis
    
    st.markdown("---")
    st.markdown("### 📊 Consolidated Nutritional Profile Dashboard Data Elements")
    
    # Grid expands natively from left to right across 100% canvas width space 
    macro_grid = payload_matrix.get("nutrition", {})
    m1, m2, m3, m4 = st.columns(4)
    with m1: components.metric_card("🔥", "Calories", macro_grid.get("calories", "N/A"))
    with m2: components.metric_card("🥩", "Protein Core", macro_grid.get("protein", "N/A"))
    with m3: components.metric_card("🧈", "Fat Profile", macro_grid.get("fat", "N/A"))
    with m4: components.metric_card("🌾", "Carbs Matrix", macro_grid.get("carbohydrates", "N/A"))
    
    # Broad full-width interactive tabs allocation panel
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["🥗 Ingredient Framework", "👨‍🍳 Cooking Recipe Steps", "🔄 Substitutions & Alternatives"])
    with t1:
        components.info_card("Ingredients Formulation Profile Matrix", payload_matrix.get("ingredients", []), "🥗")
    with t2:
        components.info_card("Step-by-step Standard Preparation Protocol Sequence", payload_matrix.get("recipe", []), "🍳")
    with t3:
        components.info_card("Optimized Low Glycemic Bio-Alternatives Recommendations", payload_matrix.get("alternatives", []), "🥦")
    
    # Dynamic Health Assessment module spans fully at the base
    st.markdown("<br>", unsafe_allow_html=True)
    components.health_score_card(
        score=float(payload_matrix.get("health_score", 7.5)),
        verdict=payload_matrix.get("verdict", "Approved Diet Component"),
        description=", ".join(payload_matrix.get("suggestions", ["Regulate overall consumption pattern variations parameters."]))
    )

# ==========================================================
# 7. FOOTER
# ==========================================================
st.write("")
st.divider()
st.markdown("""
    <center>
        <h3>🍽️ Vision-to-Nutrition</h3>
        An EfficientNet–Gemini Framework for Intelligent Food Analysis
        Powered by EfficientNetB0 + Gemini AI
    </center>
""", unsafe_allow_html=True)
