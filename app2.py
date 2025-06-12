import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import joblib
import time
from PIL import Image
import os
import logging
from typing import Optional, Tuple, List
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: #000000;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    
    .sentence-display {
        background: #000000;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        min-height: 60px;
        margin: 20px 0;
    }
    
    .stButton>button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .metric-container {
        background: black;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    
    .status-active { background-color: #4CAF50; }
    .status-inactive { background-color: #f44336; }
    .status-warning { background-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

class SignLanguageTranslator:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.hands = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        
        self.current_letter = None
        self.consecutive_count = 0
        self.threshold_frames = 15
        self.confidence_threshold = 0.7
        self.last_prediction_time = time.time()
        
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def load_models(self):
        """Load ML models and initialize MediaPipe"""
        try:
            with st.spinner("Loading AI models..."):
                if os.path.exists("isl_model (2).h5"):
                    self.model = load_model("isl_model (2).h5")
                    st.success("✅ Keras model loaded successfully")
                else:
                    st.error("❌ Keras model file not found")
                    return False
                
                # Load label encoder
                if os.path.exists("label_encoder (1).pkl"):
                    self.label_encoder = joblib.load("label_encoder (1).pkl")
                    st.success("✅ Label encoder loaded successfully")
                else:
                    st.error("❌ Label encoder file not found")
                    return False
                
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
                st.success("✅ MediaPipe initialized successfully")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            logger.error(f"Model loading error: {e}")
            return False
    
    def extract_landmarks(self, results) -> List[float]:
        """Extract hand landmarks from MediaPipe results"""
        landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        while len(landmarks) < 126:
            landmarks.append(0.0)
        
        return landmarks[:126]
    
    def predict_sign(self, landmarks: List[float]) -> Tuple[str, float]:
        """Predict sign from landmarks"""
        try:
            if len(landmarks) != 126:
                return "Unknown", 0.0
            
            input_data = np.expand_dims(np.array(landmarks), axis=0)
            input_data = np.expand_dims(input_data, axis=1)
            
            prediction = self.model.predict(input_data, verbose=0)
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            self.total_predictions += 1
            return predicted_label, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0
    
    def update_prediction_tracking(self, predicted_label: str, confidence: float):
        """Update prediction tracking and return confirmed letter"""
        confirmed_letter = None
        
        if confidence > self.confidence_threshold:
            if predicted_label == self.current_letter:
                self.consecutive_count += 1
            else:
                self.current_letter = predicted_label
                self.consecutive_count = 1
            
            if self.consecutive_count >= self.threshold_frames:
                confirmed_letter = self.current_letter
                self.consecutive_count = 0
                self.correct_predictions += 1
        else:
            self.consecutive_count = 0
            
        return confirmed_letter

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "confirmed_text" not in st.session_state:
        st.session_state.confirmed_text = ""
    if "translator" not in st.session_state:
        st.session_state.translator = SignLanguageTranslator()
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    # Add new session state for text updates
    if "text_update_queue" not in st.session_state:
        st.session_state.text_update_queue = queue.Queue()

def render_main_interface():
    """Render the main application interface"""
    st.markdown("""
    <div class="main-header">
        <h1>Unspoken:  Sign Language Translator</h1>
        <p>Real-time Indian Sign Language to Text Translation</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.models_loaded:
        if st.session_state.translator.load_models():
            st.session_state.models_loaded = True
        else:
            st.error("Failed to load required models. Please check model files.")
            return
    

    process_text_updates()
    
    main_col, control_col = st.columns([3, 1])
    
    with main_col:
        render_camera_interface()
    
    with control_col:
        render_control_panel()

def process_text_updates():
    """Process pending text updates from the queue"""
    try:
        while not st.session_state.text_update_queue.empty():
            new_letter = st.session_state.text_update_queue.get_nowait()
            st.session_state.confirmed_text += new_letter
    except queue.Empty:
        pass

def render_camera_interface():
    """Render camera interface and video processing"""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        camera_enabled = st.checkbox(
            "📹 Enable Camera",
            value=st.session_state.camera_active,
            key="camera_toggle"
        )
    
    with col2:
        show_landmarks = st.checkbox("🔍 Show Landmarks", value=True)
    
    with col3:
        mirror_mode = st.checkbox("🪞 Mirror Mode", value=True)
    
    frame_placeholder = st.empty()
    prediction_col1, prediction_col2 = st.columns([2, 1])
    
    with prediction_col1:
        current_prediction = st.empty()
    
    with prediction_col2:
        confidence_meter = st.empty()
    
    if camera_enabled and st.session_state.models_loaded:
        st.session_state.camera_active = True
        process_camera_feed(frame_placeholder, current_prediction, confidence_meter, show_landmarks, mirror_mode)
    elif camera_enabled:
        st.warning("⚠️ Models not loaded. Please wait for initialization.")
    else:
        st.session_state.camera_active = False
        frame_placeholder.info("📷 Enable camera to start sign recognition")

def process_camera_feed(frame_placeholder, prediction_display, confidence_display, show_landmarks, mirror_mode):
    """Process camera feed for sign recognition"""
    translator = st.session_state.translator
    
    if not st.session_state.get("camera_toggle", False):
        st.session_state.camera_active = False
        return
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check permissions.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        start_time = time.time()
        frame_count = 0
        
        frames_per_batch = 10
        
        while st.session_state.camera_active and st.session_state.get("camera_toggle", False):
            for batch_frame in range(frames_per_batch):
                if not st.session_state.get("camera_toggle", False):
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame")
                    break
                
                frame_count += 1
                
                if mirror_mode:
                    frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = translator.hands.process(rgb_frame)
                
                landmarks = translator.extract_landmarks(results)
                predicted_label, confidence = translator.predict_sign(landmarks)
           
                confirmed_letter = translator.update_prediction_tracking(predicted_label, confidence)

                if confirmed_letter:
                    st.session_state.text_update_queue.put(confirmed_letter)
                
                if show_landmarks and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        translator.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, translator.mp_hands.HAND_CONNECTIONS,
                            translator.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
                            translator.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                
                status_color = (0, 255, 0) if confidence > translator.confidence_threshold else (0, 255, 255)
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(frame, f'Count: {translator.consecutive_count}/{translator.threshold_frames}', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                prediction_display.markdown(f"""
                <div class="prediction-box">
                    <h4>Current Prediction: {predicted_label}</h4>
                    <p>Progress: {translator.consecutive_count}/{translator.threshold_frames}</p>
                </div>
                """, unsafe_allow_html=True)
                
                confidence_color = "🟢" if confidence > translator.confidence_threshold else "🟡"
                confidence_display.markdown(f"""
                <div class="metric-container">
                    <h4>{confidence_color} Confidence</h4>
                    <h2>{confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_fps = 30 / (time.time() - start_time)
                    start_time = time.time()
                    logger.info(f"FPS: {current_fps:.1f}")
                
                time.sleep(0.033)  
            
            process_text_updates()
            
    except Exception as e:
        st.error(f"❌ Camera processing error: {str(e)}")
        logger.error(f"Camera error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        st.session_state.camera_active = False

def render_control_panel():
    """Render control panel with text editing and settings"""
    st.markdown("###  Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("<- Backspace", key="backspace"):
            if st.session_state.confirmed_text:
                st.session_state.confirmed_text = st.session_state.confirmed_text[:-1]
                st.rerun()  
    
    with col2:
        if st.button(" Clear All", key="clear"):
            st.session_state.confirmed_text = ""
            st.rerun()  
    
    if st.button("_Add Space", key="space"):
        st.session_state.confirmed_text += " "
        st.rerun()  
    
    process_text_updates()
    
    st.markdown("### 📝 Translated Text")
    st.markdown(f"""
    <div class="sentence-display">
        {st.session_state.confirmed_text if st.session_state.confirmed_text else "Your translated text will appear here..."}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("###  Settings")
    
    if st.session_state.models_loaded:
        new_threshold = st.slider(
            "Detection Sensitivity",
            min_value=5,
            max_value=30,
            value=st.session_state.translator.threshold_frames,
            help="Lower values = more sensitive"
        )
        st.session_state.translator.threshold_frames = new_threshold
        
        new_confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.translator.confidence_threshold,
            step=0.1,
            help="Minimum confidence for predictions"
        )
        st.session_state.translator.confidence_threshold = new_confidence
    
    if st.session_state.models_loaded:
        st.markdown("### 📊 Performance")
        
        accuracy = 0
        if st.session_state.translator.total_predictions > 0:
            accuracy = st.session_state.translator.correct_predictions / st.session_state.translator.total_predictions
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", st.session_state.translator.total_predictions)
        with col2:
            st.metric("Accuracy", f"{accuracy:.1%}")

def render_help_section():
    """Render help section in sidebar"""
    st.sidebar.markdown("## ~  Help & Guide")
    
    help_section = st.sidebar.expander(" How to Use", expanded=False)
    with help_section:
        st.markdown("""
        **Getting Started:**
        1. Enable camera access
        2. Position your hand clearly in frame
        3. Make distinct sign gestures
        4. Hold signs steady for recognition
        5. Use control buttons to edit text
        
        **Tips for Better Recognition:**
        - Good lighting is essential
        - Plain background works best
        - Keep hand centered in frame
        - Make clear, distinct gestures
        - Practice consistent positioning
        """)
    
    dataset_section = st.sidebar.expander(" Sign Examples", expanded=False)
    with dataset_section:
        display_dataset_examples()
    
    troubleshooting = st.sidebar.expander(" Troubleshooting", expanded=False)
    with troubleshooting:
        st.markdown("""
        **Common Issues:**
        - **Low accuracy**: Improve lighting, use plain background
        - **Slow recognition**: Adjust sensitivity settings
        - **Camera not working**: Check browser permissions
        - **Model errors**: Ensure model files are present
        
        **Performance Tips:**
        - Close other applications using camera
        - Use good lighting conditions
        - Ensure stable internet connection
        - Practice signs clearly and consistently
        """)

def display_dataset_examples():
    """Display dataset examples if available"""
    dataset_path = "imagesss"
    
    if os.path.exists(dataset_path):
        letters = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))])
        
        if letters:
            selected_letter = st.selectbox(
                "View sign examples:",
                letters,
                key="help_letter_selector"
            )
            
            letter_path = os.path.join(dataset_path, selected_letter)
            if os.path.exists(letter_path):
                images = [f for f in os.listdir(letter_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    img_path = os.path.join(letter_path, images[0])
                    try:
                        image = Image.open(img_path)
                        image = image.resize((150, 150), Image.Resampling.LANCZOS)
                        st.image(image, caption=f"Letter {selected_letter}")
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
    else:
        st.info("Dataset folder not found. Example images not available.")

def main():
    """Main application function"""
    initialize_session_state()
    
    render_help_section()
    
    render_main_interface()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Unspoken</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

    