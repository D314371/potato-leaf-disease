import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #2E8B57; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #3CB371; font-weight: 600;}
    .metric-card {background-color: #F8F9FA; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .positive {color: #28A745;}
    .negative {color: #DC3545;}
    .model-card {border-left: 5px solid #3CB371; padding: 15px; margin: 10px 0;}
    .stProgress > div > div > div > div {background-color: #3CB371;}
</style>
""", unsafe_allow_html=True)

# Class mapping based on your training
CLASS_NAMES = {
    0: 'Potato___Early_blight',
    1: 'Potato___Late_blight', 
    2: 'Potato___healthy',
    3: 'Potato___leafroll_virus'
}

# Friendly disease names for display
FRIENDLY_NAMES = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight': 'Late Blight', 
    'Potato___healthy': 'Healthy Potato',
    'Potato___leafroll_virus': 'Leafroll Virus'
}

# Disease information
DISEASE_INFO = {
    'Potato___Early_blight': {
        'description': 'Early blight is a common fungal disease that causes dark spots with concentric rings on leaves.',
        'treatment': 'Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves and practice crop rotation.',
        'symptoms': 'Dark spots with concentric rings, yellowing leaves, premature leaf drop',
        'severity': 'Moderate'
    },
    'Potato___Late_blight': {
        'description': 'Late blight is a serious fungal disease that causes water-soaked lesions that turn brown and necrotic.',
        'treatment': 'Use fungicides with metalaxyl or famoxadone. Destroy infected plants and avoid overhead watering.',
        'symptoms': 'Water-soaked lesions, white mold under leaves, rapid plant decay',
        'severity': 'High'
    },
    'Potato___healthy': {
        'description': 'The plant appears healthy with no signs of disease.',
        'treatment': 'Continue good practices: proper spacing, balanced fertilization, and regular monitoring.',
        'symptoms': 'None',
        'severity': 'None'
    },
    'Potato___leafroll_virus': {
        'description': 'Leafroll virus causes upward curling of leaves and stunted growth, often transmitted by aphids.',
        'treatment': 'Remove infected plants. Control aphid populations with insecticides or natural predators.',
        'symptoms': 'Upward curling leaves, stunted growth, purplish discoloration',
        'severity': 'High'
    }
}

# Model performance data
MODEL_PERFORMANCE = {
    "Xception": {
        "accuracy": 0.9695,
        "precision": 0.97,
        "recall": 0.97,
        "f1": 0.97,
        "path": "Xception_model.h5",
        "size": "88 MB",
        "parameters": "22.8M",
        "description": "Xception uses depthwise separable convolutions for efficient feature extraction.",
        "confusion_matrix": [[376, 12, 8, 4], [6, 243, 1, 0], [0, 0, 261, 0], [0, 0, 0, 236]]
    },
    "InceptionV3": {
        "accuracy": 0.9712,
        "precision": 0.97,
        "recall": 0.97,
        "f1": 0.97,
        "path": "InceptionV3_model.h5",
        "size": "92 MB",
        "parameters": "23.8M",
        "description": "InceptionV3 uses parallel convolutional operations for multi-scale feature extraction.",
        "confusion_matrix": [[384, 8, 6, 2], [8, 235, 7, 0], [0, 0, 261, 0], [0, 0, 0, 236]]
    },
    "MobileNetV2": {
        "accuracy": 0.9913,
        "precision": 0.99,
        "recall": 0.99,
        "f1": 0.99,
        "path": "MobileNetV2_model.h5",
        "size": "14 MB",
        "parameters": "3.4M",
        "description": "MobileNetV2 is optimized for mobile devices with inverted residual blocks.",
        "confusion_matrix": [[392, 5, 3, 0], [3, 245, 2, 0], [0, 0, 261, 0], [0, 0, 0, 236]]
    }
}

# Class names for confusion matrix
class_names = ['Early Blight', 'Late Blight', 'Healthy', 'Leafroll Virus']

# Load the selected model
@st.cache_resource
def load_selected_model(model_name):
    try:
        model_path = MODEL_PERFORMANCE[model_name]["path"]
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_name} model: {str(e)}")
        return None

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale like in training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to create performance comparison chart
def create_performance_barchart():
    models = list(MODEL_PERFORMANCE.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Prepare data for plotting
    data = []
    for metric in metrics:
        for model in models:
            data.append({
                'Model': model,
                'Metric': metric.capitalize(),
                'Score': MODEL_PERFORMANCE[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart with Inferno color scheme
    fig = px.bar(
        df, 
        x='Model', 
        y='Score', 
        color='Metric',
        barmode='group',
        title='Model Performance Comparison by Metric',
        labels={'Score': 'Score', 'Model': 'Model'},
        color_discrete_sequence=px.colors.sequential.Inferno
    )
    
    # Format y-axis as percentage and improve styling
    fig.update_layout(
        yaxis_tickformat='.0%',
        yaxis_range=[0.9, 1.0],
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_size=20,
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    # Add value annotations on bars with better visibility
    fig.update_traces(
        texttemplate='%{y:.2%}', 
        textposition='outside',
        textfont=dict(color='white', size=12)
    )
    
    return fig

# Function to create confusion matrix heatmap
def create_confusion_matrix_heatmap(model_name):
    cm = MODEL_PERFORMANCE[model_name]["confusion_matrix"]
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        aspect="auto",
        color_continuous_scale='Inferno',
        title=f'{model_name} Confusion Matrix'
    )
    
    # Add annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(color='white' if cm[i][j] > np.max(cm)/2 else 'black')
            )
    
    # Update layout for dark theme
    fig.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_x=0.5
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown("Upload an image of a potato leaf to detect diseases using our advanced AI models.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section:", 
                               ["Home", "Disease Detection", "Model Comparison", "About"])
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Disease Detection":
        show_detection()
    elif app_mode == "Model Comparison":
        show_comparison()
    elif app_mode == "About":
        show_about()

def show_home():
    st.markdown('<p class="sub-header">Welcome to Potato Disease Detection</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        This application uses state-of-the-art deep learning models to detect and classify 
        diseases in potato plants. Early detection of plant diseases can significantly 
        reduce crop losses and improve yield quality.
        
        ### Features:
        - **Multiple Model Support**: Choose between Xception, InceptionV3, and MobileNetV2
        - **High Accuracy**: Models trained on thousands of potato leaf images
        - **Detailed Analysis**: Get disease information and treatment recommendations
        - **Performance Comparison**: Compare different model performances
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2916/2916439.png", width=150)
    
    # Quick stats
    st.markdown("### Model Performance Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Xception Accuracy", f"{MODEL_PERFORMANCE['Xception']['accuracy']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("InceptionV3 Accuracy", f"{MODEL_PERFORMANCE['InceptionV3']['accuracy']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MobileNetV2 Accuracy", f"{MODEL_PERFORMANCE['MobileNetV2']['accuracy']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance chart
    st.plotly_chart(create_performance_barchart(), use_container_width=True)

def show_detection():
    st.markdown('<p class="sub-header">Potato Disease Detection</p>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_name = st.selectbox(
            "Select Model:",
            list(MODEL_PERFORMANCE.keys()),
            help="Choose which model to use for prediction"
        )
    
    with col2:
        st.info(f"**{model_name}** - Accuracy: {MODEL_PERFORMANCE[model_name]['accuracy']*100:.2f}%")
    
    # Load the selected model
    model = load_selected_model(model_name)
    if model is None:
        st.error(f"Could not load the {model_name} model. Please check the model path.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a potato leaf image", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make prediction
            with st.spinner(f'Analyzing with {model_name}...'):
                prediction = model.predict(processed_image)
                predicted_class_idx = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                friendly_name = FRIENDLY_NAMES[predicted_class]
            
            # Display results
            st.subheader("🔍 Detection Results")
            
            # Confidence indicator
            if confidence > 0.8:
                status_emoji = "✅"
                status_color = "positive"
            elif confidence > 0.6:
                status_emoji = "⚠️"
                status_color = "negative"
            else:
                status_emoji = "❌"
                status_color = "negative"
            
            st.markdown(f"**Status:** <span class='{status_color}'>{status_emoji} {friendly_name}</span>", 
                       unsafe_allow_html=True)
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Disease information
            with st.expander("Disease Details"):
                st.write(f"**Description:** {DISEASE_INFO[predicted_class]['description']}")
                st.write(f"**Symptoms:** {DISEASE_INFO[predicted_class]['symptoms']}")
                st.write(f"**Severity:** {DISEASE_INFO[predicted_class]['severity']}")
                st.write(f"**Treatment:** {DISEASE_INFO[predicted_class]['treatment']}")
        
        # Prediction probabilities chart
        st.subheader("Prediction Probabilities")
        
        # Create a bar chart with plotly
        classes = [FRIENDLY_NAMES[CLASS_NAMES[i]] for i in range(len(CLASS_NAMES))]
        probabilities = prediction[0]
        
        fig = px.bar(
            x=classes, 
            y=probabilities,
            labels={'x': 'Disease', 'y': 'Probability'},
            color=classes,
            color_discrete_sequence=['#DC3545', '#DC3545', '#28A745', '#DC3545']
        )
        
        fig.update_layout(
            title=f'Prediction Confidence ({model_name})',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_comparison():
    st.markdown('<p class="sub-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    # Performance metrics in bar chart format
    st.plotly_chart(create_performance_barchart(), use_container_width=True)
    
    # Model details with dark cards
    st.subheader("Model Details")
    
    selected_model = st.selectbox("Select model to view details:", list(MODEL_PERFORMANCE.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Custom card styling to match Inferno theme
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #000000, #2a0845); 
                    padding: 20px; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 20px;">
            <h3 style="color: #fcba03; margin-top: 0;">{selected_model}</h3>
            <p>{MODEL_PERFORMANCE[selected_model]["description"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics with custom styling
        col1a, col2a, col3a = st.columns(3)
        with col1a:
            st.markdown(f"""
            <div style="background: #160b39; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: #fcba03; margin: 0;">Accuracy</h4>
                <h2 style="color: #fcba03; margin: 0;">{MODEL_PERFORMANCE[selected_model]['accuracy']*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2a:
            st.markdown(f"""
            <div style="background: #160b39; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: #fcba03; margin: 0;">Parameters</h4>
                <h2 style="color: #fcba03; margin: 0;">{MODEL_PERFORMANCE[selected_model]["parameters"]}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3a:
            st.markdown(f"""
            <div style="background: #160b39; padding: 15px; border-radius: 10px; text-align: center;">
                <h4 style="color: #fcba03; margin: 0;">Size</h4>
                <h2 style="color: #fcba03; margin: 0;">{MODEL_PERFORMANCE[selected_model]["size"]}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Confusion matrix with dark theme
        st.plotly_chart(create_confusion_matrix_heatmap(selected_model), use_container_width=True)
    
    # Detailed metrics table with dark styling
    st.subheader("Detailed Performance Metrics")
    
    metrics_data = []
    for model_name, metrics in MODEL_PERFORMANCE.items():
        metrics_data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Precision": f"{metrics['precision']*100:.2f}%",
            "Recall": f"{metrics['recall']*100:.2f}%",
            "F1-Score": f"{metrics['f1']*100:.2f}%"
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Apply dark styling to the dataframe
    st.markdown("""
    <style>
    .dataframe {
        background-color: #160b39;
        color: white;
    }
    .dataframe th {
        background-color: #fcba03;
        color: black;
    }
    .dataframe tr:nth-child(even) {
        background-color: #2a0845;
    }
    .dataframe tr:nth-child(odd) {
        background-color: #160b39;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(df, use_container_width=True)

def show_about():
    st.markdown('<p class="sub-header">About This Application</p>', unsafe_allow_html=True)
    
    st.write("""
    This Potato Disease Detection application uses deep learning models to identify 
    common diseases in potato plants from leaf images. The models were trained on 
    a comprehensive dataset containing thousands of images across four categories:
    
    - **Early Blight**: Fungal disease causing concentric ring spots
    - **Late Blight**: Serious fungal disease causing water-soaked lesions
    - **Leafroll Virus**: Viral disease causing upward curling of leaves
    - **Healthy**: Disease-free plants
    
    ### Technology Stack
    - **Framework**: TensorFlow, Keras
    - **Models**: Xception, InceptionV3, MobileNetV2
    - **Frontend**: Streamlit
    - **Visualization**: Plotly, Matplotlib
    
    ### Model Performance
    All models achieved excellent performance, with MobileNetV2 leading at 99.13% accuracy,
    demonstrating that lightweight models can achieve state-of-the-art results in plant
    disease detection.
    
    ### Applications
    This technology can be deployed in:
    - Precision agriculture systems
    - Mobile applications for farmers
    - Automated farming equipment
    - Research and educational tools
    """)

if __name__ == "__main__":
    main()