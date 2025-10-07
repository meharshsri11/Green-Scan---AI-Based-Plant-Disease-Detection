import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import tempfile
import os
from fpdf import FPDF
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Green-Scan",
    page_icon="🌿",
    layout="centered"
)

# --- INITIALIZE SESSION STATE FOR HISTORY ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MODEL AND DATA DEFINITIONS ---
@st.cache_resource
def load_my_model():
    """Loads the trained model once and caches it."""
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# List of all class names the model can predict
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- FULLY POPULATED REMEDY INFORMATION ---
REMEDY_INFO = {
    "Apple___Apple_scab": "Remove and destroy infected leaves. Apply a fungicide in the spring.",
    "Apple___Black_rot": "Prune out and destroy cankered branches. Apply a fungicide during the growing season.",
    "Apple___Cedar_apple_rust": "Apply fungicides from pink bud stage until petal fall. Remove nearby juniper trees if possible.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply fungicides (such as sulfur or potassium bicarbonate). Ensure good air circulation by pruning.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids. Practice crop rotation and tillage to reduce fungal residue.",
    "Corn_(maize)___Common_rust_": "Plant resistant hybrids. Fungicides can be effective if applied early.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and practice crop rotation. Fungicides can be effective if applied early.",
    "Grape___Black_rot": "Remove and destroy infected canes and leaves. Apply fungicides from early shoot growth until berries ripen.",
    "Grape___Esca_(Black_Measles)": "There is no definitive cure. Prune out and destroy infected wood during the dormant season to manage the disease.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Generally minor. Ensure good air circulation. Fungicides for other grape diseases typically control it.",
    "Orange___Haunglongbing_(Citrus_greening)": "There is no cure. Prevention is key: control the psyllid insect vector and remove infected trees immediately.",
    "Peach___Bacterial_spot": "Plant resistant varieties. Apply bactericides (copper-based products) during the dormant season.",
    "Pepper,_bell___Bacterial_spot": "Use disease-free seed and transplants. Practice crop rotation. Copper sprays can help manage spread.",
    "Potato___Early_blight": "Apply fungicides preventively. Practice crop rotation and ensure proper plant nutrition.",
    "Potato___Late_blight": "Apply fungicides preventively, especially during cool, moist conditions. Destroy infected plants.",
    "Squash___Powdery_mildew": "Improve air circulation. Apply fungicides like neem oil or sulfur. Avoid overhead watering.",
    "Strawberry___Leaf_scorch": "Remove infected leaves after harvest. Ensure proper spacing for air circulation. Use resistant varieties.",
    "Tomato___Bacterial_spot": "Use disease-free seeds. Avoid overhead watering. Copper-based bactericides can help.",
    "Tomato___Early_blight": "Prune plants to improve air circulation. Mulch the soil. Apply a fungicide containing copper.",
    "Tomato___Late_blight": "Apply fungicides preventively. Ensure proper spacing and remove infected plants.",
    "Tomato___Leaf_Mold": "Ensure proper ventilation and lower humidity. Prune lower leaves. Apply a fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves. Mulch around the base of plants. Apply a fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "This is a pest. Spray with insecticidal soap or neem oil, covering the undersides of leaves.",
    "Tomato___Target_Spot": "Practice crop rotation. Improve air circulation. Apply fungicides preventively during warm, humid weather.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly populations with insecticides or netting. Remove and destroy infected plants.",
    "Tomato___Tomato_mosaic_virus": "There is no cure. Remove and destroy infected plants. Wash hands and tools to prevent spread."
}

# --- FULLY POPULATED YIELD IMPACT INFORMATION ---
YIELD_IMPACT_INFO = {
    "Apple___Apple_scab": "Primarily affects fruit quality. Can cause 10-20% yield loss in severe cases.",
    "Apple___Black_rot": "Can cause significant fruit rot, leading to 25-50% yield loss if unmanaged.",
    "Apple___Cedar_apple_rust": "Minor impact on yield, but can cause cosmetic damage to fruit and defoliation.",
    "Cherry_(including_sour)___Powdery_mildew": "Can reduce fruit quality and size. Severe infections may cause yield losses of 15-25%.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Yield loss is typically minor (5-15%) but can be higher in susceptible hybrids.",
    "Corn_(maize)___Common_rust_": "Generally causes minimal yield loss (less than 10%) unless infection is very early and severe.",
    "Corn_(maize)___Northern_Leaf_Blight": "Can cause yield losses of 15-30% if it infects the plant before flowering.",
    "Grape___Black_rot": "Can cause complete crop loss if not controlled, as it rots the fruit clusters.",
    "Grape___Esca_(Black_Measles)": "Chronic disease that can lead to vine death, resulting in total yield loss for that plant.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Typically has a negligible impact on yield unless leaf drop is severe.",
    "Orange___Haunglongbing_(Citrus_greening)": "Devastating. Infected trees produce small, bitter fruit and die, leading to total yield loss.",
    "Peach___Bacterial_spot": "Reduces fruit quality and can cause premature fruit drop, leading to yield losses of 10-30%.",
    "Pepper,_bell___Bacterial_spot": "Can cause significant defoliation and fruit spots, leading to yield losses of 20-50%.",
    "Potato___Early_blight": "Can cause defoliation that reduces tuber size, leading to yield losses of 20-30%.",
    "Potato___Late_blight": "One of the most destructive diseases; can cause up to 75% or even total crop loss.",
    "Squash___Powdery_mildew": "Reduces plant vigor, leading to smaller and fewer fruits, with potential yield loss of 20-40%.",
    "Strawberry___Leaf_scorch": "Can weaken plants, reducing runner production and yield by 10-25% in severe cases.",
    "Tomato___Bacterial_spot": "Can cause significant defoliation and fruit lesions, leading to 30-50% yield loss.",
    "Tomato___Early_blight": "Can cause a yield reduction of 20-40% if not managed.",
    "Tomato___Late_blight": "Extremely destructive, can lead to total crop loss in a short period.",
    "Tomato___Leaf_Mold": "Reduces photosynthetic area, which can weaken the plant and reduce overall yield by 10-20%.",
    "Tomato___Septoria_leaf_spot": "Causes significant defoliation, weakening the plant. Can reduce yield by 20-40%.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Heavy infestations can cause leaves to die, significantly stressing the plant and reducing yield by up to 50%.",
    "Tomato___Target_Spot": "Can cause fruit rot and defoliation, leading to yield losses of 15-30%.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Stunts plant growth and prevents fruit production, often leading to 90-100% yield loss.",
    "Tomato___Tomato_mosaic_virus": "Reduces fruit quality and quantity, with yield losses typically ranging from 10-25%."
}

# --- HELPER FUNCTIONS ---
def model_prediction(image_path, model):
    """Prepares image and returns prediction index and confidence."""
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    confidence = np.max(predictions)
    result_index = np.argmax(predictions)
    return result_index, confidence

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a Page", ["Home", "Disease Recognition", "Supported Diseases", "About"])

# --- PREDICTION HISTORY IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.header("Prediction History")
if not st.session_state.history:
    st.sidebar.write("No predictions made yet.")
else:
    for item in st.session_state.history[-5:]:
        st.sidebar.write(f"**• {item['name']}** ({item['confidence']:.2f}%)")

# --- PAGE CONTENT ---

if app_mode == "Home":
    st.title("Green-Scan - AI-Based Plant Disease Detection")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("home_page.jpeg")
    st.markdown("""
    ### Welcome to Green-Scan! 🌿
    Your smart solution for instant plant disease diagnosis. Green-Scan leverages advanced Artificial Intelligence to analyze leaf images and provide accurate, real-time results, helping you protect your crops and ensure a healthier harvest.
    
    ---
    
    ### How It Works
    1.  **Submit an Image:** Navigate to the **Disease Recognition** page and either upload a leaf image from your device or paste a direct URL.
    2.  **Instant AI Analysis:** Our deep learning model processes the image, identifying key visual patterns to detect signs of disease.
    3.  **Get Your Diagnosis:** Receive an accurate diagnosis in seconds, helping you take timely and effective action.
    
    ---
    
    ### Key Features
    -   **High Accuracy:** Powered by a state-of-the-art Convolutional Neural Network (CNN) for reliable predictions.
    -   **User-Friendly Interface:** A clean, simple, and intuitive design that makes disease detection effortless.
    -   **Rapid Results:** Go from image to diagnosis in just a few seconds.
    """)

elif app_mode == "Disease Recognition":
    st.header("Get Your Diagnosis")
    st.markdown("---")
    
    image_path_for_prediction = None
    option = st.radio("Select Input Method:", ("Upload an Image", "Provide Image URL"))

    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path_for_prediction = tmp_file.name

    elif option == "Provide Image URL":
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption='Image from URL', use_column_width=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    img.save(tmp_file.name)
                    image_path_for_prediction = tmp_file.name
            except Exception as e:
                st.error("Error loading image from URL. Please check the link.")

    if st.button("Predict"):
        model = load_my_model()
        if image_path_for_prediction is not None and model is not None:
            with st.spinner("Analyzing..."):
                result_index, confidence = model_prediction(image_path_for_prediction, model)
                
                predicted_class = CLASS_NAMES[result_index]
                display_name = predicted_class.replace('___', ' - ').replace('_', ' ')
                
                st.success(f"**Diagnosis:** {display_name}")
                st.info(f"**Confidence:** {confidence*100:.2f}%")
                
                remedy = REMEDY_INFO.get(predicted_class, "No remedy information available for this specific condition.")
                yield_impact = YIELD_IMPACT_INFO.get(predicted_class, "No yield impact data available for this specific condition.")

                if "healthy" in predicted_class:
                    st.balloons()
                    st.info("The plant appears to be healthy! No specific action is required.")
                else:
                    with st.expander("Show Recommended Action"):
                        st.write(remedy)
                    with st.expander("Show Potential Yield Impact"):
                        st.write(yield_impact)

                    st.session_state.history.append({"name": display_name, "confidence": confidence * 100})
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, "Green-Scan Diagnosis Report", 0, 1, 'C')
                    pdf.set_font("Arial", '', 12)
                    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
                    pdf.ln(10)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Diagnosis:", 0, 1)
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, f"{display_name} (Confidence: {confidence*100:.2f}%)")
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Recommended Action:", 0, 1)
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, remedy)
                    pdf.ln(5)
                    
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "Potential Yield Impact:", 0, 1)
                    pdf.set_font("Arial", '', 12)
                    pdf.multi_cell(0, 10, yield_impact)
                    
                    pdf_output = pdf.output(dest='S')
                    
                    st.download_button(
                        label="Download Report as PDF",
                        data=bytes(pdf_output),
                        file_name=f"Green-Scan_Report_{display_name.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

            if os.path.exists(image_path_for_prediction):
                os.remove(image_path_for_prediction)
        else:
            st.error("Please upload an image or provide a valid URL first.")

elif app_mode == "Supported Diseases":
    st.header("Supported Plants & Diseases")
    st.markdown("---")
    st.write("The Green-Scan model is trained to recognize the following 38 classes:")
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, 19):
            st.text(f"• {CLASS_NAMES[i].replace('___', ' - ').replace('_', ' ')}")
    with col2:
        for i in range(19, 38):
            st.text(f"• {CLASS_NAMES[i].replace('___', ' - ').replace('_', ' ')}")

elif app_mode == "About":
    st.header("About Green-Scan")
    st.markdown("---")
    st.markdown("""
    ### Project Vision
    The mission of Green-Scan is to democratize precision agriculture. We believe that powerful AI tools shouldn't be limited to large-scale operations. By providing a free, simple, and accurate disease detection tool, we aim to empower farmers and gardeners to make timely decisions, reduce crop loss, and promote sustainable farming practices.

    ---

    ### How It Works: The Technology
    Green-Scan is a web application built entirely in Python, using several key technologies:

    * **Streamlit:** For creating the interactive and user-friendly web interface you're using right now.
    * **TensorFlow & Keras:** For building and running the core of our system—a deep learning model known as a Convolutional Neural Network (CNN).
    * **The Model:** Our CNN was trained on the extensive PlantVillage dataset, which contains over 87,000 images. It has learned to identify the unique visual patterns and textures of 38 different plant diseases and healthy leaves.

    ---

    ### Key Features
    This application has been developed with the user in mind, incorporating features like:

    * **Dual Image Input:** You can either upload a file directly from your device or paste a URL for instant analysis.
    * **Instant Diagnosis:** Get a prediction in seconds, powered by our trained AI model.
    * **Actionable Advice:** For diagnosed diseases, the app provides recommended actions and remedies to help you treat your plants.
    * **Supported Diseases List:** A comprehensive list of all diseases the model can currently identify, so you know what to expect.
    """)