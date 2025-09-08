ASL Voice Translator
A real-time American Sign Language (ASL) letter translator that uses your webcam to recognize hand gestures, build sentences, and convert them to speech using AI.

Live Demo
(It's highly recommended to record a short GIF of your application in action and place it here. This is the best way to showcase your project!)

[Your GIF demonstrating the translator in action]

Features
Real-Time Hand Tracking: Utilizes Google's MediaPipe to automatically detect and track your hand anywhere in the frame.

Letter-to-Sentence Engine: Translates individual ASL letter signs and intelligently assembles them into complete sentences.

Voice Output: Speaks the completed sentence aloud when you pause, using Google's Text-to-Speech for clear, natural-sounding audio.

Polished User Interface: A clean, semi-transparent UI panel displays the live camera feed, current prediction, action feedback, and the sentence being built.

Gesture Commands:

Delete: Use the 'B' sign to function as a backspace.

Space: Use the 'V' sign to add a space between words.

Clear: Use the 'Y' sign to clear the entire sentence and start fresh.

Screenshots
Main Interface in Action

Hand Detection with MediaPipe

(Place your screenshot here)

(Place your screenshot here)

A view of the app recognizing a letter.

A view showing the landmarks drawn on your hand.

Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
You will need to have the following software installed:

Python 3.9+

Git

Installation & Setup
Clone the repository (after you've pushed it to GitHub):

git clone [https://github.com/YOUR_USERNAME/ASL-Voice-Translator.git](https://github.com/YOUR_USERNAME/ASL-Voice-Translator.git)
cd ASL-Voice-Translator

Create and Activate a Virtual Environment:

# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
# source venv/bin/activate

Install Dependencies:
This project uses a requirements.txt file to manage all necessary libraries. Install them with this one command:

pip install -r requirements.txt

How to Use the Translator
Run the application:

python run_translator.py

Start Signing:

A window will open showing your webcam feed.

Hold your hand in front of the camera. The app will automatically detect it and draw landmarks.

Begin signing ASL letters. The predicted letter will appear in the UI panel and be added to the sentence.

Use Commands for Editing:

Sign 'B' to delete the last character.

Sign 'V' to add a space.

Sign 'Y' to clear the current sentence.

Hear the Voice Output:

When you finish a word or sentence, simply pause for about 2.5 seconds.

The application will automatically speak the full sentence you have built.

Quit the Application:

To close the program, make sure the webcam window is active and press the 'q' key.