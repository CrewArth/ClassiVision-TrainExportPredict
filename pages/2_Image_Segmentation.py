import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import io
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import colorsys

# Set page configuration
st.set_page_config(
    page_title="ClassiVision - Image Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS
st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load DeepLabV3+ model"""
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for the model"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def generate_distinct_colors(n):
    """Generate n visually distinct colors using improved color space sampling"""
    colors = []
    
    # Base hues for common objects (road, sky, vegetation, buildings, etc.)
    base_hues = [
        0.6,   # Blue for sky
        0.3,   # Green for vegetation
        0.15,  # Orange/Brown for buildings
        0.0,   # Red for vehicles
        0.7,   # Purple for infrastructure
        0.45,  # Teal for water
    ]
    
    # Add base colors first
    for hue in base_hues[:min(n, len(base_hues))]:
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(np.array(rgb) * 255)
    
    # Generate additional colors if needed
    if n > len(base_hues):
        remaining = n - len(base_hues)
        for i in range(remaining):
            # Use golden ratio to space out hues evenly
            hue = (i * 0.618033988749895) % 1
            # Vary saturation and value slightly for more distinction
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.1
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(np.array(rgb) * 255)
    
    return colors

def detect_edges(image):
    """Advanced edge detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges for better visibility
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    return dilated

def process_image(image, model):
    """Process image through DeepLabV3+ model"""
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Get predictions
    predictions = output.argmax(0).cpu().numpy()
    
    return predictions

def create_colored_mask(predictions, edges):
    """Create colored segmentation mask with improved color assignment"""
    # Count actual number of classes in the predictions
    unique_classes = np.unique(predictions)
    num_classes = len(unique_classes)
    
    # Generate colors with improved distinction
    colors = generate_distinct_colors(num_classes)
    
    # Create colored mask
    colored_mask = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    
    # Create a mapping of class indices to ensure consistent coloring
    class_to_color_idx = {class_idx: i for i, class_idx in enumerate(unique_classes)}
    
    # Fill regions with colors
    for class_idx in unique_classes:
        mask = predictions == class_idx
        color_idx = class_to_color_idx[class_idx]
        colored_mask[mask] = colors[color_idx]
    
    # Apply edge-aware smoothing with reduced intensity
    colored_mask = cv2.bilateralFilter(colored_mask, 7, 50, 50)
    
    return colored_mask

def apply_post_processing(colored_mask):
    """Apply advanced post-processing for enhanced segmentation visualization"""
    # Initial bilateral filter with reduced intensity
    smoothed = cv2.bilateralFilter(colored_mask, 7, 75, 75)
    
    # Enhance contrast and color saturation
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance lightness channel with reduced clipping
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance color channels with increased saturation
    a = cv2.multiply(a, 1.3)  # Increased color saturation
    b = cv2.multiply(b, 1.3)  # Increased color saturation
    
    # Merge channels
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Final light smoothing
    final = cv2.bilateralFilter(enhanced, 5, 40, 40)
    
    return final

def main():
    st.title("🎯 Advanced Semantic Segmentation")
    st.markdown("Segment objects in images with high precision")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            # Read and process image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Create columns for display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("### Segmentation Result")
                with st.spinner("Processing..."):
                    # Detect edges
                    edges = detect_edges(image)
                    
                    # Process image through model
                    predictions = process_image(image, model)
                    
                    # Create colored mask with edge preservation
                    colored_mask = create_colored_mask(predictions, edges)
                    
                    # Apply post-processing
                    enhanced_mask = apply_post_processing(colored_mask)
                    
                    # Display result
                    st.image(enhanced_mask, use_column_width=True)
                    
                    # Add download button
                    buf = io.BytesIO()
                    result_img = Image.fromarray(enhanced_mask)
                    result_img.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Segmentation",
                        data=buf,
                        file_name="segmentation_result.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            # Display blend of original and segmentation
            st.markdown("### Blended Result")
            alpha = st.slider("Blend Ratio", 0.0, 1.0, 0.5)
            
            # Convert PIL Image to numpy array
            original_array = np.array(image)
            
            # Blend images
            blended = cv2.addWeighted(
                original_array, 
                1 - alpha,
                enhanced_mask, 
                alpha, 
                0
            )
            
            # Create a smaller version of the blended image
            max_width = 400  # Maximum width for the blended result
            aspect_ratio = blended.shape[0] / blended.shape[1]
            new_width = min(max_width, blended.shape[1])
            new_height = int(new_width * aspect_ratio)
            blended_resized = cv2.resize(blended, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Display resized blended result
            st.image(blended_resized, use_column_width=False)
            
            # Add download button for blended result
            buf = io.BytesIO()
            blend_img = Image.fromarray(blended)  # Save full resolution for download
            blend_img.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="Download Blended Result",
                data=buf,
                file_name="blended_result.png",
                mime="image/png",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
