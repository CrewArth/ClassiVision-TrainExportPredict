import cv2
import time
import os
import streamlit as st
from PIL import Image

def try_capture_device(index, backend=None):
    """Helper function to try different capture methods"""
    try:
        if backend:
            cap = cv2.VideoCapture(index, backend)
        else:
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass
    return None

def capture_frames(class_path, interval=2, max_frames=30):
    """
    Captures frames from webcam at specified intervals
    
    Args:
        class_path: Directory to save captured frames
        interval: Time between captures in seconds
        max_frames: Maximum number of frames to capture
    """
    # Create directory if it doesn't exist
    os.makedirs(class_path, exist_ok=True)
    
    # Try different camera configurations
    cap = None
    backends = [
        (0, cv2.CAP_DSHOW),  # Try Windows DirectShow first
        (0, cv2.CAP_MSMF),   # Try Windows Media Foundation
        (0, None),           # Try default backend
        (1, cv2.CAP_DSHOW),  # Try second camera with DirectShow
        (1, None),           # Try second camera with default backend
    ]
    
    for index, backend in backends:
        cap = try_capture_device(index, backend)
        if cap is not None:
            break
    
    if cap is None or not cap.isOpened():
        st.error("""Could not access webcam. Please try the following:
1. Make sure your webcam is connected
2. Close other applications that might be using the webcam
3. Try unplugging and plugging back in your webcam
4. Check if your webcam works in Camera app or other applications""")
        return []
    
    # Test if we can actually read from the camera
    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        st.error("Connected to webcam but couldn't read video stream. Please check your webcam drivers.")
        return []
    
    captured_frames = []
    frame_count = 0
    last_capture = time.time()
    
    # Create webcam container with animated border
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    progress_placeholder = st.empty()
    counter_placeholder = st.markdown(
        f'<div class="webcam-overlay">Captured: 0/{max_frames}</div>',
        unsafe_allow_html=True
    )
    
    stop_button = st.button("Stop Capture", key="stop_webcam")
    
    while frame_count < max_frames and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update progress bar
        progress = frame_count / max_frames
        progress_placeholder.progress(progress)
        
        # Capture frame if interval has passed
        current_time = time.time()
        if current_time - last_capture >= interval:
            # Generate unique filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = os.path.join(class_path, f"frame_{timestamp}_{frame_count:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            captured_frames.append(frame_path)
            frame_count += 1
            last_capture = current_time
            
            # Update counter with animation
            counter_placeholder.markdown(
                f'<div class="webcam-overlay" style="animation: pulse 0.5s ease">Captured: {frame_count}/{max_frames}</div>',
                unsafe_allow_html=True
            )
            
            # Show preview grid of last 8 captured frames
            if len(captured_frames) > 0:
                st.markdown('<div class="sample-grid">', unsafe_allow_html=True)
                for img_path in captured_frames[-8:]:  # Show last 8 frames
                    img = Image.open(img_path)
                    st.image(img, width=100, caption=f"Frame {frame_count}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a small delay to prevent overloading
        time.sleep(0.1)
    
    # Clean up
    cap.release()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Final status
    if frame_count > 0:
        st.success(f"Successfully captured {frame_count} frames!")
    else:
        st.warning("No frames were captured.")
    
    return captured_frames
