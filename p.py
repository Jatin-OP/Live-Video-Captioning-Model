import streamlit as st
import cv2
import time
import torch
import pyttsx3
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Generation settings
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to predict caption
def predict_caption(frame):
    """Preprocess frame and predict caption."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    return caption

# Function to speak captions
def speak_caption(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Live Video Captioning with Voice Output")
st.markdown("Captions will appear above the video and will be spoken aloud.")

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

frame_interval = 5  # seconds
last_capture_time = time.time()
captions = st.empty()  # Placeholder for captions

# Stream video feed in Streamlit
stframe = st.empty()  # Placeholder for video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame.")
        break

    # Convert frame to RGB (Streamlit needs RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show video feed in Streamlit
    stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    # Capture frame every 5 seconds
    if time.time() - last_capture_time >= frame_interval:
        caption = predict_caption(frame)
        captions.markdown(f"### Caption: {caption}")  # Display caption above video
        speak_caption(caption)  # Speak caption
        last_capture_time = time.time()  # Reset timer

    # Break on 'q' key (not applicable in Streamlit, but useful for debugging)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
