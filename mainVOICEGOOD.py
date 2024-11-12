import sounddevice as sd
import numpy as np
import threading
import queue
import time
from openai import OpenAI
import wave
import os
import tempfile

#pip install sounddevice numpy openai wave os tempfile PySoundFile scipy react


class EmergencyDispatcher:
    def __init__(self):
        # Initialize OpenAI client
        self.client = OpenAI(api_key="OPENAPIKEYHERE")
        self.assistant_id = "asst_DGcJujd3wtjBRZ4KsdrD0q5X"
        self.thread = self.client.beta.threads.create()
        
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 0.05  # 100ms chunks for speech detection
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Speech detection parameters
        self.speech_threshold = 700  # Adjust based on your microphone
        self.silence_duration = 1.5  # Seconds of silence to end recording
        self.min_audio_length = 0.5  # Minimum audio length to process
        self.speech_frames = []
        self.silence_frames = 0
        self.is_recording = False
        
        # State management
        self.call_in_progress = True
        self.temp_dir = tempfile.mkdtemp()

    def detect_speech(self, audio_data):
        """Detect if audio contains speech using amplitude threshold."""
        return np.abs(audio_data).mean() > self.speech_threshold

    def record_and_process(self):
        """Continuously record and process audio with speech detection."""
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            
            # Check for speech in current chunk
            if self.detect_speech(indata):
                if not self.is_recording:
                    print("Speech detected - starting recording...")
                    self.is_recording = True
                self.speech_frames.append(indata.copy())
                self.silence_frames = 0
            elif self.is_recording:
                self.silence_frames += 1
                self.speech_frames.append(indata.copy())  # Keep some silence for natural speech
                
                # Check if silence duration exceeded
                silence_time = self.silence_frames * self.chunk_duration
                if silence_time >= self.silence_duration:
                    print("Silence detected - processing speech...")
                    self.process_recorded_speech()
                    self.is_recording = False
                    self.speech_frames = []
                    self.silence_frames = 0

        try:
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                callback=audio_callback,
                dtype=np.int16
            ):
                print("Listening for speech...")
                while self.call_in_progress:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio stream: {e}")

    def process_recorded_speech(self):
        """Process the recorded speech frames."""
        if not self.speech_frames:
            return

        try:
            # Combine all frames
            audio_data = np.concatenate(self.speech_frames)
            duration = len(audio_data) / self.sample_rate

            # Only process if audio is long enough
            if duration >= self.min_audio_length:
                # Save to temporary WAV file
                temp_path = os.path.join(self.temp_dir, f"speech_{time.time()}.wav")
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())

                # Transcribe
                with open(temp_path, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )

                os.remove(temp_path)

                if transcript and transcript.strip():
                    print(f"Caller: {transcript}")
                    self.handle_input(transcript)

        except Exception as e:
            print(f"Error processing recorded speech: {e}")

    def text_to_speech(self, text):
        """Convert text to speech using OpenAI's TTS."""
        try:
            temp_path = os.path.join(self.temp_dir, f"response_{time.time()}.mp3")
            
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="shimmer",
                input=text
            )
            
            response.stream_to_file(temp_path)
            
            if os.path.exists(temp_path):
                if os.name == 'posix':
                    os.system(f"afplay '{temp_path}'")
                elif os.name == 'nt':
                    os.system(f'start "" "{temp_path}"')
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Text-to-speech error: {e}")

    def handle_input(self, text):
        """Handle transcribed input and get AI response."""
        if not text:
            return

        try:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=text
            )

            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id
            )

            start_time = time.time()
            while time.time() - start_time < 30:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    messages = self.client.beta.threads.messages.list(
                        thread_id=self.thread.id
                    )
                    
                    for msg in messages.data:
                        if msg.role == "assistant":
                            response = msg.content[0].text.value
                            print(f"Dispatcher: {response}")
                            self.text_to_speech(response)
                            return
                time.sleep(0.5)

        except Exception as e:
            print(f"Error handling input: {e}")
            self.text_to_speech("I'm experiencing technical difficulties. Please hold.")

    def run(self):
        """Main method to run the dispatcher."""
        try:
            # Initial greeting
            self.text_to_speech("911, what's your emergency?")
            
            # Start recording and processing
            self.record_and_process()

        except KeyboardInterrupt:
            print("\nEmergency dispatcher shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.call_in_progress = False
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {e}")

def main():
    dispatcher = EmergencyDispatcher()
    try:
        dispatcher.run()
    except KeyboardInterrupt:
        print("\nEmergency dispatcher terminated by operator.")
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        dispatcher.cleanup()


if __name__ == "__main__":
    main()