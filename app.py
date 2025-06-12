import gradio as gr
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import joblib
import time
import threading
from queue import Queue

# Global variables for optimization
frame_queue = Queue(maxsize=2)  # Limit queue size to reduce lag
processing_active = False

# Load models once
# @gr.utils.cached_model
def load_models():
    try:
        model = load_model("isl_model (2).h5")
        label_encoder = joblib.load("label_encoder (1).pkl")
        return model, label_encoder, True
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, False

model, label_encoder, models_loaded = load_models()

# Initialize MediaPipe with optimized settings
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,  # Reduced to 1 for better performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # Use lighter model
)

class OptimizedSignLanguageState:
    def __init__(self):
        self.current_letter = None
        self.consecutive_count = 0
        self.confirmed_text = ""
        self.threshold_frames = 8  # Reduced for faster response
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # Limit predictions to 10 FPS

state = OptimizedSignLanguageState()

def extract_landmarks_fast(results):
    """Optimized landmark extraction"""
    if not results.multi_hand_landmarks:
        return []
    
    landmarks = []
    # Only process first hand for speed
    hand_landmarks = results.multi_hand_landmarks[0]
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def predict_sign_fast(landmarks):
    """Optimized prediction with caching"""
    if not models_loaded or len(landmarks) < 63:
        return "None"
    
    try:
        # Pad landmarks if needed
        if len(landmarks) < 126:
            landmarks.extend([0] * (126 - len(landmarks)))
        
        # Use only first 63 landmarks for faster processing if needed
        input_data = np.expand_dims(np.array(landmarks[:126]), axis=0)
        input_data = np.expand_dims(input_data, axis=1)
        
        # Predict with batch_size=1 for speed
        prediction = model.predict(input_data, verbose=0, batch_size=1)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        return predicted_label
    except Exception as e:
        return "Error"

def process_video_optimized(frame):
    """Highly optimized video processing"""
    global processing_active
    
    if frame is None or processing_active:
        return frame, "Processing...", state.confirmed_text
    
    processing_active = True
    
    try:
        current_time = time.time()
        
        # Skip processing if too frequent
        if current_time - state.last_prediction_time < state.prediction_interval:
            processing_active = False
            return frame, f"Current: {state.current_letter or 'None'} ({state.consecutive_count}/{state.threshold_frames})", state.confirmed_text
        
        state.last_prediction_time = current_time
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640.0 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Extract landmarks
        landmarks = extract_landmarks_fast(results)
        
        # Predict sign
        predicted_label = predict_sign_fast(landmarks)
        
        # Draw landmarks (simplified)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Only first hand
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
        
        # Update state
        if predicted_label == state.current_letter:
            state.consecutive_count += 1
        else:
            state.current_letter = predicted_label
            state.consecutive_count = 1
        
        # Confirm letter
        if (state.consecutive_count >= state.threshold_frames and 
            predicted_label not in ["None", "Error"]):
            state.confirmed_text += predicted_label
            state.consecutive_count = 0
        
        # Add text overlays (simplified)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Current: {predicted_label}', (10, 30),
                    font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Count: {state.consecutive_count}/{state.threshold_frames}', (10, 60),
                    font, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f'Text: {state.confirmed_text[-12:]}', (10, 90),
                    font, 0.5, (255, 255, 0), 1)
        
        processing_active = False
        return frame, f"Current: {predicted_label} ({state.consecutive_count}/{state.threshold_frames})", state.confirmed_text
        
    except Exception as e:
        processing_active = False
        return frame, f"Error: {str(e)[:20]}", state.confirmed_text

def clear_text():
    state.confirmed_text = ""
    return ""

def backspace_text():
    state.confirmed_text = state.confirmed_text[:-1]
    return state.confirmed_text

def add_space():
    state.confirmed_text += " "
    return state.confirmed_text

def reset_recognition():
    state.current_letter = None
    state.consecutive_count = 0
    return "Recognition reset"

def update_threshold(value):
    state.threshold_frames = int(value)
    return f"Threshold updated to {value}"

# Custom CSS for better performance
css = """
.gradio-container {
    max-width: 1200px !important;
}
#video_input {
    max-height: 500px;
}
.output-image {
    max-height: 500px !important;
}
"""

with gr.Blocks(title="Optimized Sign Language Translator", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("""
    # 🤟 Optimized Sign Language Translator
    
    **Real-time Indian Sign Language recognition - Optimized for Performance**
    
    ### Performance Optimizations:
    - Reduced processing frequency (10 FPS instead of 30)
    - Single hand detection for speed
    - Lighter MediaPipe model
    - Frame resizing for faster processing
    - Simplified drawing operations
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Image(
                sources=["webcam"], 
                type="numpy",
                streaming=True,
                mirror_webcam=False,
                elem_id="video_input"
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                backspace_btn = gr.Button("⬅️ Back", variant="secondary", size="sm") 
                space_btn = gr.Button("␣ Space", variant="secondary", size="sm")
                reset_btn = gr.Button("🔄 Reset", variant="secondary", size="sm")
            
            # Performance settings
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=3, maximum=15, value=8, step=1,
                    label="Confirmation Frames (Lower = Faster)",
                    interactive=True
                )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Status")
            status_output = gr.Textbox(
                label="Recognition Status", 
                interactive=False,
                max_lines=2
            )
            
            gr.Markdown("### 📝 Text Output")
            text_output = gr.Textbox(
                label="Translated Text", 
                placeholder="Letters will appear here...",
                interactive=False,
                lines=6
            )
            
            # Performance indicator
            performance_info = gr.Textbox(
                label="Performance Info",
                value="Optimized for low-lag processing",
                interactive=False,
                max_lines=1
            )
            
            if not models_loaded:
                gr.Markdown("""
                ### ⚠️ Models Not Loaded
                Ensure files exist:
                - `isl_model (2).h5`
                - `label_encoder (1).pkl`
                """)

            gr.Markdown("""
            ### 💡 Performance Tips
            
            **For Best Performance:**
            - Lower confirmation frames (3-5) for faster response
            - Use good lighting to help detection
            - Keep hand movements steady
            - Close other resource-heavy applications
            
            **If Still Lagging:**
            - Try the Streamlit version instead
            - Reduce browser window size
            - Use Chrome for better performance
            """)
    
    # Event handlers with optimized settings
    video_input.stream(
        fn=process_video_optimized,
        inputs=[video_input],
        outputs=[video_input, status_output, text_output],
        stream_every=0.1,  # 10 FPS instead of default
        show_progress=False
    )
    
    # Button handlers
    clear_btn.click(fn=clear_text, outputs=[text_output])
    backspace_btn.click(fn=backspace_text, outputs=[text_output])
    space_btn.click(fn=add_space, outputs=[text_output])
    reset_btn.click(fn=reset_recognition, outputs=[status_output])
    threshold_slider.change(fn=update_threshold, inputs=[threshold_slider], outputs=[performance_info])

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_port=7864,
        inbrowser=True,
        show_error=True,
        max_threads=2
    )