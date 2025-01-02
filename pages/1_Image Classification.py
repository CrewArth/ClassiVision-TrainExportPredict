import streamlit as st
import os
import shutil
import cv2
import time
import zipfile
import io
from PIL import Image
from utils.model_utils import train_model, predict_image, plot_training
from utils.webcam_utils import capture_frames

# Page config
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if 'classes' not in st.session_state:
    st.session_state['classes'] = []
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'show_advanced' not in st.session_state:
    st.session_state['show_advanced'] = False

# Constants
DATA_DIR = "data/"
MODEL_DIR = "models/"
os.makedirs(DATA_DIR, exist_ok=True)

# Custom CSS
st.markdown("""
<style>
    /* Fix navbar overlap */
    .main {
        padding-top: 3rem !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }
    
    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.5rem;
    }
    .stButton button {
        width: 100%;
        padding: 0.3rem;
        margin: 0;
    }
    
    /* Header adjustments */
    h3 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact class container */
    .class-container {
        border: 1px solid #444;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        background: rgba(0,0,0,0.2);
    }
    .class-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
    }
    .class-name {
        font-size: 1rem;
        font-weight: bold;
    }
    .sample-count {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Training stats */
    .training-stats {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        background: rgba(0,0,0,0.2);
        border-radius: 0.5rem;
        padding: 0.3rem;
        text-align: center;
        flex: 1;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Compact divider */
    .section-divider {
        margin: 0.5rem 0;
        border-color: #444;
    }
    
    /* Hide preview during capture */
    .stVideo {
        display: none;
    }
    
    /* Improve radio button layout */
    .row-widget.stRadio > div {
        flex-direction: row !important;
        margin-bottom: 0.5rem;
    }
    
    /* Adjust file uploader */
    .stFileUploader {
        padding: 0 !important;
        margin-bottom: 0.5rem;
    }
    
    .uploadedFile {
        padding: 0.3rem !important;
    }
    
    /* Custom Upload Box */
    .css-1cpxqw2 {
        background-color: rgba(35, 35, 35, 0.8) !important;
        border: 2px dashed rgba(128, 128, 128, 0.5) !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 5px 0 !important;
    }
    
    .css-1cpxqw2:hover {
        border-color: #ffffff !important;
        background-color: rgba(45, 45, 45, 0.8) !important;
    }
    
    /* Upload Text */
    .css-1cpxqw2 p {
        color: #ffffff !important;
        font-size: 0.9rem !important;
    }
    
    /* File Type Text */
    .css-1cpxqw2 small {
        color: #a0a0a0 !important;
        font-size: 0.8rem !important;
    }
    
    /* Browse Files Button */
    .css-1cpxqw2 button {
        background-color: rgba(70, 70, 70, 0.8) !important;
        border: 1px solid rgba(128, 128, 128, 0.5) !important;
        color: #ffffff !important;
        padding: 0.3rem 1rem !important;
        font-size: 0.9rem !important;
        margin-top: 0.5rem !important;
        border-radius: 4px !important;
    }
    
    .css-1cpxqw2 button:hover {
        background-color: rgba(90, 90, 90, 0.8) !important;
        border-color: #ffffff !important;
    }
    
    /* Uploaded Files List */
    .css-16j7yh3 {
        background-color: rgba(35, 35, 35, 0.8) !important;
        border-radius: 4px !important;
        margin: 2px 0 !important;
        padding: 0.3rem !important;
    }
    
    /* Progress Bar */
    .css-16j7yh3 progress {
        height: 6px !important;
        margin: 4px 0 !important;
    }
    
    /* File Name */
    .css-16j7yh3 p {
        color: #ffffff !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    
    /* Make expander more compact */
    .streamlit-expanderHeader {
        padding: 0.5rem !important;
    }
    .streamlit-expanderContent {
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Main layout - Three columns
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.markdown("### 1. Manage Classes")
    
    # Add Class Section
    col_input, col_add, col_delete = st.columns([2, 1, 1])
    
    with col_input:
        new_class = st.text_input("", placeholder="Enter class name", label_visibility="collapsed")
    
    with col_add:
        if st.button("➕ Add Class", use_container_width=True):
            if new_class and new_class not in st.session_state['classes']:
                class_path = os.path.join(DATA_DIR, new_class)
                os.makedirs(class_path, exist_ok=True)
                st.session_state['classes'].append(new_class)
                st.toast(f"Added class: {new_class}")
                time.sleep(0.5)
                st.rerun()
    
    with col_delete:
        if st.button("🗑️ Reset All", type="secondary", use_container_width=True):
            try:
                # Delete all class data
                if os.path.exists(DATA_DIR):
                    shutil.rmtree(DATA_DIR)
                os.makedirs(DATA_DIR, exist_ok=True)
                
                # Delete model files if they exist
                if os.path.exists(MODEL_DIR):
                    shutil.rmtree(MODEL_DIR)
                os.makedirs(MODEL_DIR, exist_ok=True)
                
                # Reset session state
                st.session_state['classes'] = []
                st.session_state['model_trained'] = False
                if 'training_stats' in st.session_state:
                    del st.session_state['training_stats']
                
                st.toast("All data and models have been reset!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reset: {str(e)}")
    
    # Display Classes in scrollable container
    with st.container():
        for class_name in st.session_state['classes']:
            class_path = os.path.join(DATA_DIR, class_name)
            image_count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            st.markdown(f"""
            <div class="class-container">
                <div class="class-header">
                    <span class="class-name">{class_name}</span>
                    <span class="sample-count">{image_count} samples</span>
                </div>
            """, unsafe_allow_html=True)
            
            col_webcam, col_upload, col_delete = st.columns([1, 1, 0.5])
            
            with col_webcam:
                if st.button("📷 Webcam", key=f"webcam_{class_name}", help="Capture from Webcam"):
                    st.session_state['current_class'] = class_name
                    with st.spinner("Capturing..."):
                        captured_images = capture_frames(class_path)
                        if captured_images:
                            st.toast(f"Captured {len(captured_images)} images!")
            
            with col_upload:
                if st.button("📤 Upload", key=f"upload_btn_{class_name}", help="Upload Images", use_container_width=True):
                    st.session_state[f'show_uploader_{class_name}'] = True
                
                if st.session_state.get(f'show_uploader_{class_name}', False):
                    uploaded_files = st.file_uploader(
                        "Drop images here",
                        accept_multiple_files=True,
                        type=["jpg", "jpeg", "png"],
                        key=f"uploader_{class_name}",
                        label_visibility="collapsed",
                        help="Drag and drop your images here or click to browse"
                    )
                    if uploaded_files:
                        with st.spinner(f"Uploading images for {class_name}..."):
                            for file in uploaded_files:
                                with open(os.path.join(class_path, file.name), 'wb') as f:
                                    f.write(file.getvalue())
                            st.toast(f"Uploaded {len(uploaded_files)} images!")
                            st.session_state[f'show_uploader_{class_name}'] = False
                            time.sleep(0.5)
                            st.rerun()
            
            with col_delete:
                if st.button("Delete", key=f"delete_{class_name}", help="Delete class data"):
                    try:
                        shutil.rmtree(class_path)
                        os.makedirs(class_path, exist_ok=True)
                        st.toast(f"Deleted data for: {class_name}")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 2. Model Training")
    
    # Create columns for training controls
    col_model, col_epochs, col_batch = st.columns(3)
    
    with col_model:
        model_name = st.selectbox(
            "Select Model",
            ["VGG16", "ResNet50", "MobileNet", "EfficientNetB0"],
            help="Choose the model architecture for training"
        )
    
    with col_epochs:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=25)
    
    with col_batch:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=32)
    
    # Advanced options expander
    with st.expander("Advanced Options"):
        col_lr, col_dropout = st.columns(2)
        with col_lr:
            learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=0.0001, format="%f")
        with col_dropout:
            dropout_rate = st.number_input("Dropout Rate", min_value=0.0, max_value=0.9, value=0.3)
        
        col_aug, col_stop = st.columns(2)
        with col_aug:
            data_augmentation = st.checkbox("Data Augmentation", value=True)
        with col_stop:
            early_stopping = st.checkbox("Early Stopping", value=True)
    
    # Training button and progress
    if st.button("Start Training", use_container_width=True):
        if len(st.session_state['classes']) < 2:
            st.error("Please add at least 2 classes before training")
        else:
            with st.spinner("Training model..."):
                # Create containers for stats
                stats_container = st.empty()
                progress_bar = st.progress(0)
                
                # Store containers in session state
                st.session_state['stats_container'] = stats_container
                st.session_state['progress_bar'] = progress_bar
                
                try:
                    # Train model with selected parameters
                    results = train_model(
                        model_name=model_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        dropout_rate=dropout_rate,
                        data_augmentation=data_augmentation,
                        early_stopping=early_stopping
                    )
                    
                    if results:
                        st.success("Training completed successfully!")
                        st.session_state['model_trained'] = True
                        
                        # Plot training history
                        plot = plot_training()
                        if plot:
                            st.pyplot(plot)
                    else:
                        st.error("Training failed. Please try again.")
                
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                
                finally:
                    # Clear progress indicators
                    progress_bar.empty()
                    stats_container.empty()

    # Download Section
    if st.session_state['model_trained']:
        st.markdown("---")
        st.markdown("#### Export Model")
        
        # Create a zip file containing the model and class indices
        if st.button("📦 Download Model", use_container_width=True):
            try:
                with st.spinner("Preparing download..."):
                    # Create a BytesIO object to store the zip file
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        # Add model file
                        model_path = os.path.join(MODEL_DIR, "trained_model.keras")
                        if os.path.exists(model_path):
                            zip_file.write(model_path, "model/trained_model.keras")
                        
                        # Add class indices
                        indices_path = os.path.join(MODEL_DIR, "class_indices.npy")
                        if os.path.exists(indices_path):
                            zip_file.write(indices_path, "model/class_indices.npy")
                        
                        # Add training history
                        history_path = os.path.join(MODEL_DIR, "training_history.npy")
                        if os.path.exists(history_path):
                            zip_file.write(history_path, "model/training_history.npy")
                        
                        # Add a README file
                        readme_content = f"""# ClassiVision Model Export

This package contains:
1. Trained model (trained_model.keras)
2. Class indices (class_indices.npy)
3. Training history (training_history.npy)

Model Information:
- Number of classes: {len(st.session_state['classes'])}
- Classes: {', '.join(st.session_state['classes'])}
- Training accuracy: {st.session_state['training_stats'].get('accuracy', 0):.2%}
- Input size: {224}x{224} pixels
"""
                        zip_file.writestr("README.md", readme_content)
                    
                    # Prepare the zip file for download
                    zip_buffer.seek(0)
                    
                    # Create download button
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="classivision_model.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    st.success("Model package prepared successfully!")
            except Exception as e:
                st.error(f"Failed to prepare download: {str(e)}")

with col3:
    st.markdown("### 3. Prediction")
    
    if st.session_state['model_trained']:
        test_method = st.radio("Test Method:", ["Webcam", "Upload"], horizontal=True)
        
        if test_method == "Webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam")
            else:
                # Create placeholders
                frame_placeholder = st.empty()
                prediction_placeholder = st.empty()
                confidence_placeholder = st.empty()
                
                # Create stop button
                stop_btn = st.button("⏹️ Stop Prediction")
                
                try:
                    while not stop_btn:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame")
                            break
                        
                        # Convert frame from BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Display the frame
                        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Make prediction
                        # Convert frame to PIL Image
                        pil_image = Image.fromarray(frame_rgb)
                        prediction, confidence = predict_image(pil_image)
                        
                        if prediction and confidence > 0.7:
                            prediction_placeholder.success(f"Prediction: {prediction}")
                            confidence_placeholder.info(f"Confidence: {confidence:.2%}")
                        else:
                            prediction_placeholder.warning("No confident prediction")
                            if prediction:
                                confidence_placeholder.info(f"Confidence: {confidence:.2%}")
                        
                        # Add a small delay
                        time.sleep(0.1)
                
                finally:
                    cap.release()
                    # Clear placeholders
                    frame_placeholder.empty()
                    prediction_placeholder.empty()
                    confidence_placeholder.empty()
        
        else:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                # Create columns for preview and prediction
                col_preview, col_prediction = st.columns([1, 1])
                
                with col_preview:
                    # Load and resize image for preview
                    image = Image.open(uploaded_file)
                    # Calculate height maintaining aspect ratio
                    aspect_ratio = image.size[1] / image.size[0]
                    preview_width = 300  # Fixed width for preview
                    preview_height = int(preview_width * aspect_ratio)
                    image_resized = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
                    st.image(image_resized, caption="Uploaded Image", use_column_width=True)
                
                with col_prediction:
                    # Make prediction
                    prediction, confidence = predict_image(uploaded_file)
                    if prediction and confidence > 0.7:
                        st.success(f"Prediction: {prediction}")
                        st.info(f"Confidence: {confidence:.2%}")
                    else:
                        st.warning(f"Low confidence prediction: {prediction}")
                        st.info(f"Confidence: {confidence:.2%}")
    else:
        st.info("Train a model first")

def capture_frames(class_path):
    """Capture frames from webcam with mobile support"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam")
        return []
    
    captured_images = []
    
    # Create columns for capture and switch camera buttons
    col1, col2 = st.columns(2)
    
    with col1:
        capture_btn = st.button("📸 Capture", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)
    
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    
    try:
        while not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            if capture_btn:
                # Create a unique filename
                timestamp = int(time.time() * 1000)
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(class_path, filename)
                
                # Save the captured frame
                cv2.imwrite(filepath, frame)
                captured_images.append(filepath)
                st.success(f"Image captured successfully!")
                break
            
            # Add a small delay
            time.sleep(0.1)
    
    finally:
        cap.release()
    
    return captured_images



