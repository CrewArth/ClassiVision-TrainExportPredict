import streamlit as st

# Page configuration
st.set_page_config(
    page_title="ClassiVision - Train, Predict & Effortless Export",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--background-color);
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 1rem;
        background: var(--background-color);
        border-bottom: 1px solid var(--primary-color);
        position: sticky;
        top: 0;
        z-index: 999;
        color: var(--text-color);
    }
    
    /* Hero Section */
    .hero {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(45deg, rgba(var(--primary-color-rgb), 0.1), rgba(var(--secondary-color-rgb), 0.1));
        border-radius: 20px;
        margin: 2rem 0;
        color: var(--text-color);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: var(--text-color);
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Feature Section */
    .feature-section {
        background: var(--background-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid var(--primary-color);
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .feature-card {
        background: var(--background-color);
        border-radius: 16px;
        padding: 2rem;
        height: 100%;
        border: 1px solid var(--primary-color);
        transition: all 0.3s ease;
        color: var(--text-color);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--primary-color);
        box-shadow: 0 5px 15px rgba(var(--primary-color-rgb), 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--secondary-color);
        margin-bottom: 1rem;
    }
    
    .feature-text {
        color: var(--text-color);
        line-height: 1.6;
    }
    
    /* Steps Section */
    .step-card {
        background: var(--background-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
        color: var(--text-color);
    }
    
    .step-number {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    /* About Section */
    .about-section {
        background: var(--background-color);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid var(--primary-color);
        color: var(--text-color);
    }
    
    .profile-img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        margin-bottom: 1.5rem;
        border: 3px solid var(--primary-color);
    }
    
    /* Contact Section */
    .contact-section {
        background: var(--background-color);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid var(--primary-color);
        color: var(--text-color);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: var(--background-color);
        margin-top: 3rem;
        border-top: 1px solid var(--primary-color);
        color: var(--text-color);
    }
    
    /* Buttons */
    .cta-button {
        display: inline-block;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 30px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        margin: 0.5rem;
    }
    
    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(var(--primary-color-rgb), 0.3);
        opacity: 0.9;
    }
    
    /* Theme Variables */
    :root {
        --primary-color: #4ECDC4;
        --primary-color-rgb: 78, 205, 196;
        --secondary-color: #FF6B6B;
        --secondary-color-rgb: 255, 107, 107;
    }
    
    /* Light theme */
    [data-theme="light"] {
        --background-color: #ffffff;
        --text-color: #1e293b;
        --border-color: rgba(0, 0, 0, 0.1);
    }
    
    /* Dark theme */
    [data-theme="dark"] {
        --background-color: #0f172a;
        --text-color: #ffffff;
        --border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1.2rem;
        }
        .section-title {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add theme detection script
st.markdown("""
<script>
    const theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
</script>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🤖 ClassiVision - Train, Predict & Effortless Export</h1>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-title">Train AI Models with Ease</div>
    <div class="hero-subtitle">
        Build powerful image classification and segmentation models without writing a single line of code
    </div>
    <a href="Image_Classification" class="cta-button">Get Started →</a>
</div>
""", unsafe_allow_html=True)

# Features Section
st.markdown('<div class="section-title">🚀 Key Features</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📸</div>
        <div class="feature-title">Image Classification</div>
        <div class="feature-text">
            Train custom models to classify images into different categories with high accuracy. Perfect for sorting images, identifying objects, and automating visual tasks.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Image Segmentation</div>
        <div class="feature-text">
            Segment objects in images with state-of-the-art precision using Meta's SAM model. Ideal for detecting objects, creating masks, and analyzing image components.
        </div>
    </div>
    """, unsafe_allow_html=True)

# How to Use Image Classification
st.markdown('<div class="section-title">📸 How to Use Image Classification</div>', unsafe_allow_html=True)
st.markdown("""
<div class="feature-section">
    <div class="step-card">
        <div class="step-number">Step 1: Create Classes</div>
        <div class="feature-text">Add classes for your image categories and start collecting data using webcam or file upload.</div>
    </div>
    <div class="step-card">
        <div class="step-number">Step 2: Train Model</div>
        <div class="feature-text">Configure training parameters and let ClassiVision train a custom model on your data.</div>
    </div>
    <div class="step-card">
        <div class="step-number">Step 3: Test & Export</div>
        <div class="feature-text">Test your model with new images and export it for future use.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# How to Use Image Segmentation
st.markdown('<div class="section-title">🎯 How to Use Image Segmentation</div>', unsafe_allow_html=True)
st.markdown("""
<div class="feature-section">
    <div class="step-card">
        <div class="step-number">Step 1: Upload Image</div>
        <div class="feature-text">Upload the image you want to segment using the file uploader.</div>
    </div>
    <div class="step-card">
        <div class="step-number">Step 2: Select Objects</div>
        <div class="feature-text">Click on objects in the image to generate precise segmentation masks.</div>
    </div>
    <div class="step-card">
        <div class="step-number">Step 3: Export Results</div>
        <div class="feature-text">Download the segmented masks and processed images.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# About Developer
st.markdown('<div class="section-title">👨‍💻 About Developer</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-section">
    <img src="https://avatars.githubusercontent.com/u/105881935?v=4" alt="Developer" class="profile-img">
    <h2>Arth Vala</h2>
    <p>AI/ML Engineer passionate about making machine learning accessible to everyone. With expertise in computer vision and deep learning, I created ClassiVision to help users build powerful AI models without coding.</p>
</div>
""", unsafe_allow_html=True)

# Contact Section
st.markdown('<div class="section-title">📬 Contact Us</div>', unsafe_allow_html=True)
st.markdown("""
<div class="contact-section">
    <p>Have questions or suggestions? Reach out to us!</p>
    <a href="mailto:arthvala@gmail.com" class="cta-button">Email Us</a>
    <a href="https://github.com/CrewArth" class="cta-button">GitHub</a>
    <a href="https://linkedin.com/in/arthvala" class="cta-button">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 ClassiVision. Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)


