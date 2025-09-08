# run_translator.py
# This is the main application file. It uses a hand tracking model to find the hand,
# then uses the trained sign language model to detect ASL signs, assembles them 
# into sentences, and converts them to speech.
#
# --- Phase 8: UI/UX Enhancement ---
#
# Instructions:
# 1. This version introduces a cleaner, more professional user interface.
# 2. Make sure you have successfully run 'train_model.py' and have
#    the 'asl_model.h5' file in your project directory.
# 3. Make sure all libraries are installed:
#    pip install mediapipe gTTS playsound==1.2.2
# 4. Run this script from your command prompt:
#    python run_translator.py
#
# Press 'q' to quit.

import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import queue
from gtts import gTTS
from playsound import playsound
import os
import mediapipe as mp # Import MediaPipe

# --- Setup a thread-safe queue for TTS requests ---
speech_queue = queue.Queue()

# --- Dedicated Speaker Thread Function (using gTTS) ---
def speaker_thread_func():
    # Announce that the system is ready
    try:
        print("Speaker thread: Generating 'System ready' audio...")
        tts = gTTS(text='System ready', lang='en')
        startup_file = "startup.mp3"
        tts.save(startup_file)
        print("Speaker thread: Playing 'System ready' audio...")
        playsound(startup_file)
        os.remove(startup_file)
        print("Speaker thread: 'System ready' audio finished.")
    except Exception as e:
        print(f"Error during startup sound: {e}")
        print("Please ensure you have an internet connection for gTTS to work.")

    while True:
        try:
            text_to_speak = speech_queue.get()
            if text_to_speak is None: # A way to signal the thread to exit
                print("Speaker thread: Received exit signal.")
                break
            
            speech_file = "temp_speech.mp3"
            try:
                print(f"Speaker thread: Generating audio for '{text_to_speak}'...")
                tts = gTTS(text=text_to_speak, lang='en', slow=False)
                tts.save(speech_file)
                print(f"Speaker thread: Playing audio for '{text_to_speak}'...")
                playsound(speech_file)
            finally:
                if os.path.exists(speech_file):
                    os.remove(speech_file)
            
            speech_queue.task_done()
        except Exception as e:
            print(f"Error in speaker thread: {e}")
            print("Please ensure you have a stable internet connection.")

# --- Start the Speaker Thread ---
speaker_thread = threading.Thread(target=speaker_thread_func, daemon=True)
speaker_thread.start()

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model('asl_model.h5')
    print("Model asl_model.h5 loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'asl_model.h5' is in the same directory.")
    exit()

# --- Define Labels ---
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', '?', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Real-Time Detection ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Variables for Sentence Logic & UI ---
current_sentence = []
last_letter_time = time.time()
LETTER_COOLDOWN = 1.5
last_sign_time = time.time()
SENTENCE_TIMEOUT = 2.5
CONFIDENCE_LEVEL = 0.90
action_text = ""
ui_status_text = "LISTENING..."

print("Starting video capture...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a more natural mirror-like view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    display_text = "Prediction: None"
    text_color = (255, 255, 255)
    action_text = "" # Reset action text each frame

    # --- Hand Detection and ROI Extraction ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, c = frame.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
        
        padding = 30
        x1, y1 = max(0, x_min - padding), max(0, y_min - padding)
        x2, y2 = min(w, x_max + padding), min(h, y_max + padding)
        
        # Draw dynamic bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized_roi = cv2.resize(gray_roi, (28, 28))
            normalized_roi = resized_roi / 255.0
            input_data = normalized_roi.reshape(1, 28, 28, 1)

            prediction = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_letter = labels[predicted_index]
            confidence = np.max(prediction)

            display_text = f"Prediction: {predicted_letter} (Conf: {confidence:.2f})"
            
            current_time = time.time()
            is_confident = confidence > CONFIDENCE_LEVEL

            if is_confident:
                last_sign_time = current_time
                ui_status_text = "LISTENING..."

                if (current_time - last_letter_time) > LETTER_COOLDOWN:
                    if predicted_letter == 'B':
                        action_text = "DELETE"
                        if len(current_sentence) > 0:
                            current_sentence.pop()
                    elif predicted_letter == 'V':
                        action_text = "SPACE"
                        if len(current_sentence) > 0 and current_sentence[-1] != ' ':
                             current_sentence.append(' ')
                    else:
                        current_sentence.append(predicted_letter)
                    
                    last_letter_time = current_time
    
    # Speak sentence if there's a pause
    current_time = time.time()
    if len(current_sentence) > 0 and (current_time - last_sign_time) > SENTENCE_TIMEOUT:
        full_sentence = "".join(current_sentence).replace('?','')
        ui_status_text = f"SPEAKING: {full_sentence}"
        speech_queue.put(full_sentence)
        current_sentence = []
    
    # --- UI Display ---
    # Create a semi-transparent panel at the bottom
    overlay = frame.copy()
    panel_y_start = frame.shape[0] - 120
    cv2.rectangle(overlay, (0, panel_y_start), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Display Status
    cv2.putText(frame, f"STATUS: {ui_status_text}", (20, panel_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display Prediction
    cv2.putText(frame, display_text, (20, panel_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    # Display Action
    if action_text:
        cv2.putText(frame, f"ACTION: {action_text}", (250, panel_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display Sentence
    sentence_display_text = f"Sentence: {''.join(current_sentence)}"
    cv2.putText(frame, sentence_display_text, (20, panel_y_start + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('ASL Voice Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)

