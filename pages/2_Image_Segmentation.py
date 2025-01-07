import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# Set page configuration
st.set_page_config(
    page_title="ClassiVision - Image Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto"
)

# Define Cityscapes color map
CITYSCAPES_COLORMAP = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}

@st.cache_resource
def load_model():
    """Load DeepLabV3+ model pretrained on Cityscapes"""
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

def create_colored_mask(predictions):
    """Create colored segmentation mask using Cityscapes colors"""
    # Get class predictions
    pred_mask = predictions.argmax(0).cpu().numpy()
    
    # Create colored mask
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Apply colors for each class
    for class_idx, color in CITYSCAPES_COLORMAP.items():
        colored_mask[pred_mask == class_idx] = color
    
    return colored_mask

def apply_post_processing(colored_mask):
    """Apply post-processing for enhanced visualization"""
    # Apply edge-aware smoothing
    processed = cv2.bilateralFilter(colored_mask, 9, 75, 75)
    
    # Enhance contrast
    lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def main():
    st.title("🎯 Image Segmentation")
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
                st.image(image, use_container_width =True)
            
            with col2:
                st.markdown("### Segmentation Result")
                with st.spinner("Processing..."):
                    # Preprocess image
                    input_tensor = preprocess_image(image)
                    
                    # Get predictions
                    with torch.no_grad():
                        output = model(input_tensor)['out'][0]
                    
                    # Create colored mask
                    colored_mask = create_colored_mask(output)
                    
                    # Apply post-processing
                    enhanced_mask = apply_post_processing(colored_mask)
                    
                    # Display result
                    st.image(enhanced_mask, use_container_width =True)
                    
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
            
            # Resize enhanced_mask to match original image size
            enhanced_mask_resized = cv2.resize(
                enhanced_mask, 
                (original_array.shape[1], original_array.shape[0]), 
                interpolation=cv2.INTER_AREA
            )
            
            # Blend images
            blended = cv2.addWeighted(
                original_array, 
                1 - alpha,
                enhanced_mask_resized, 
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
            st.image(blended_resized, use_container_width =False)
            
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
