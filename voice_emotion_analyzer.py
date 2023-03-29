import os
import io
import pyaudio
import queue
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import language_v1

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# Set up the speech-to-text client
speech_client = speech.SpeechClient()

# Set up the natural language client
language_client = language_v1.LanguageServiceClient()

# Set up the audio stream
audio_format = pyaudio.paInt16
channels = 1
rate = 16000
chunk = 4096

audio = pyaudio.PyAudio()

def transcribe_audio_stream(stream):
    requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in stream)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=False)
    responses = speech_client.streaming_recognize(streaming_config, requests)

    for response in responses:
        for result in response.results:
            return result.alternatives[0].transcript
    return None

def analyze_sentiment(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment_response = language_client.analyze_sentiment(request={'document': document})
    return sentiment_response.document_sentiment

def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def sentiment_to_emotion(score, magnitude):
    if -1 <= score < -0.6:
        return "Very Negative"
    elif -0.6 <= score < -0.2:
        return "Negative"
    elif -0.2 <= score < 0.2:
        return "Neutral"
    elif 0.2 <= score < 0.6:
        return "Positive"
    else: # 0.6 <= score <= 1
        return "Very Positive"

def is_silent(audio_data, threshold=1500):
    """Returns True if the audio data is below the specified threshold."""
    return np.abs(np.frombuffer(audio_data, dtype=np.int16)).mean() < threshold

def main():
    print("Starting voice recording. Speak into the microphone...")

    # Open the microphone stream with a callback function
    stream = audio.open(format=audio_format, channels=channels, rate=rate, input=True,
                        frames_per_buffer=chunk, stream_callback=callback)
    stream.start_stream()

    silent_chunks = 0
    required_silent_chunks = int(2 * rate / chunk)  # 2 seconds of silence

    audio_data = []

    try:
        while True:
            try:
                # Record audio
                data = audio_queue.get()
                audio_data.append(data)

                if is_silent(data):
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks >= required_silent_chunks:
                    audio_buffer = io.BytesIO(b"".join(audio_data[:-required_silent_chunks]))

                    # Transcribe the audio
                    audio_buffer.seek(0)
                    transcribed_text = transcribe_audio_stream(audio_buffer)
                    print(f"Transcribed text: {transcribed_text}")

                    # Analyze sentiment
                    sentiment = analyze_sentiment(transcribed_text)
                    print(f"Sentiment score: {sentiment.score}, Sentiment magnitude: {sentiment.magnitude}")
                    print(sentiment_to_emotion(sentiment.score, sentiment.magnitude))

                    audio_data = []

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    finally:
        # Close the microphone stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    audio_queue = queue.Queue()
    main()
