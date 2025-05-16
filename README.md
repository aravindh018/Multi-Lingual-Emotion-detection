# Multi-Lingual-Emotion-detection
Multilingual Emotion Detection in Voice – README
Project Overview

This project implements a multilingual speech emotion recognition system using Python. It combines automatic speech recognition (ASR) and emotion classification to analyze audio files, transcribe their content, detect the spoken language, and predict the underlying emotion. The system is designed to work with multiple languages and leverages deep learning and machine learning techniques for robust performance.

Key Features

Speech Recognition: Uses OpenAI Whisper for accurate transcription and language detection from audio files.

Emotion Recognition: Extracts audio features (MFCCs and pitch) and classifies emotions using a Support Vector Machine (SVM).

Multilingual Support: Capable of processing audio in various languages, making it suitable for global applications.

Audio Conversion: Converts MP4 audio files to WAV format for processing.

End-to-End Pipeline: From raw audio input to emotion and language output, all steps are automated.

Dependencies

The following Python packages are required:

openai-whisper

librosa

scikit-learn

numpy

moviepy

torch

All additional dependencies for the above libraries (e.g., soundfile, numba, etc.)

Install them using pip:

python
pip install openai-whisper librosa scikit-learn moviepy torch
How It Works

Audio Conversion:
The pipeline starts by converting an MP4 file to WAV format using MoviePy.

Speech Transcription:
The WAV file is transcribed using Whisper, which also detects the spoken language.

Feature Extraction:
The system extracts Mel Frequency Cepstral Coefficients (MFCCs) and pitch features from the audio using Librosa. These features are statistically summarized (mean values) and concatenated to form a feature vector.

Emotion Classification:
A Support Vector Machine (SVM) classifier, trained on labeled data, predicts the emotion from the extracted features. The classifier supports emotions such as 'happy', 'sad', 'angry', and 'neutral'.

Output:
The system prints the detected emotion, transcription, and language.

Example Usage

Replace /content/happy voice.mp4 with your actual audio file path:

python
audio_path = "/content/happy voice.mp4"
emotion_aware_speech_recognition(audio_path)
Sample Output:

text
MoviePy - Writing audio in /content/temp_audio.wav
MoviePy - Done.
Transcription: Hello how are you? Good morning everyone have a nice day
Detected Emotion: happy
Transcription: Hello how are you? Good morning everyone have a nice day
Language Detected: en
Main Functions & Pipeline

convert_mp4_to_wav(mp4_path, wav_path): Converts MP4 to WAV.

transcribe_audio(audio_path): Transcribes speech and detects language.

extract_audio_features(audio_path): Extracts MFCC and pitch features.

train_emotion_classifier(): Trains the SVM classifier (uses placeholder random data; in practice, train with real labeled datasets).

classify_emotion(features, classifier, le): Predicts emotion from features.

emotion_aware_speech_recognition(mp4_path): Complete pipeline from audio file to emotion, transcription, and language output.

Notes

The provided classifier is trained on randomly generated data for demonstration. For real-world use, train the SVM with a curated, labeled dataset containing diverse emotional speech samples in multiple languages.

The system leverages MFCCs and pitch, which are widely used features for speech emotion recognition.

You can expand the list of supported emotions and languages by using more comprehensive datasets and retraining the classifier.

For best results, ensure your audio files are clear and of good quality.

References

The methodology aligns with state-of-the-art research in multilingual speech emotion recognition, utilizing MFCCs, SVMs, and curated datasets for each language.

For more details on multilingual emotion recognition and its challenges, see the referenced academic literature.

