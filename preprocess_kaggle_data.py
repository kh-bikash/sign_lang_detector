# run_translator.py
# This is the main application file. It uses the trained model to detect
# ASL signs in real-time from your webcam, assembles them into sentences,
# and converts them to speech.
#
# --- Phase 5: Sentence Formation ---
#
# Instructions:
# 1. Make sure you have successfully run 'train_model.py' and have
#    the 'asl_model.h5' file in your project directory.
# 2. Make sure you have installed the required libraries:
#    pip install gTTS playsound==1.2.2
# 3. Run this script from your command prompt:
#    python run_translator.py
#
# A window will open showing your webcam. Form signs one by one to build a
# sentence (e.g., H-E-L-L-O). Pause for a moment when you are done,
# and the application will speak the full sentence. Press 'q' to quit.

import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import queue
# --- NEW IMPORTS for Google Text-to-Speech ---
from gtts import gTTS
from playsound import playsound
import os

# --- Setup a thread-safe queue for TTS requests ---
speech_queue = queue.Queue()

# --- Dedicated Speaker Thread Function (using gTTS) ---
# This function runs in a separate, long-lived thread.
# It now uses gTTS to create and play an MP3 file for a full sentence.
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
            
            # --- gTTS and playsound logic ---
            speech_file = "temp_speech.mp3"
            try:
                print(f"Speaker thread: Generating audio for '{text_to_speak}'...")
                tts = gTTS(text=text_to_speak, lang='en', slow=False)
                tts.save(speech_file)
                print(f"Speaker thread: Playing audio for '{text_to_speak}'...")
                playsound(speech_file)
            finally:
                # Ensure the temporary file is always deleted
                if os.path.exists(speech_file):
                    os.remove(speech_file)
            
            speech_queue.task_done()
        except Exception as e:
            print(f"Error in speaker thread: {e}")
            print("Please ensure you have a stable internet connection.")


# --- Start the Speaker Thread ---
# Set as a daemon thread so it automatically exits when the main program does
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


# --- Real-Time Detection ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Variables for Sentence Logic ---
current_sentence = []
last_letter_time = time.time()
LETTER_COOLDOWN = 1.0  # Seconds before adding the next letter
last_sign_time = time.time()  # To detect the end of a sentence
SENTENCE_TIMEOUT = 2.5 # Seconds of no signs before speaking
CONFIDENCE_LEVEL = 0.90

print("Starting video capture...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a more natural mirror-like view
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) where the hand should be
    x1, y1 = 100, 100
    x2, y2 = 400, 400
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI for the model
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (28, 28))
    normalized_roi = resized_roi / 255.0
    input_data = normalized_roi.reshape(1, 28, 28, 1)

    # --- Make Prediction ---
    prediction = model.predict(input_data, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_letter = labels[predicted_index]
    confidence = np.max(prediction)

    # --- Display Prediction on Screen ---
    display_text = f"Prediction: {predicted_letter} (Conf: {confidence:.2f})"
    text_color = (36,255,12) # Green if not confident

    # --- Sentence Formation & Text-to-Speech Logic ---
    current_time = time.time()
    is_confident = confidence > CONFIDENCE_LEVEL

    if is_confident:
        text_color = (0, 255, 255) # Yellow if confident
        # Update the time we last saw a confident sign
        last_sign_time = current_time

        # Check if enough time has passed since the last letter was added
        if (current_time - last_letter_time) > LETTER_COOLDOWN:
            current_sentence.append(predicted_letter)
            last_letter_time = current_time
            print(f"Appended letter: {predicted_letter}. Current sentence: {''.join(current_sentence)}")

    # If there's a pause and we have a sentence built up, speak it.
    if len(current_sentence) > 0 and (current_time - last_sign_time) > SENTENCE_TIMEOUT:
        full_sentence = "".join(current_sentence)
        print(f"Sentence complete due to pause. Queueing: '{full_sentence}'")
        speech_queue.put(full_sentence)
        # Reset sentence
        current_sentence = []
    
    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # --- Display the sentence being built ---
    sentence_display_text = f"Sentence: {''.join(current_sentence)}"
    cv2.putText(frame, sentence_display_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Show the final frame
    cv2.imshow('ASL Voice Translator', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
speech_queue.put(None) # Signal the speaker thread to exit

