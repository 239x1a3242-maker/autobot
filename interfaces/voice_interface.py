"""
Voice Interface for AutoBot
Handles speech-to-text and text-to-speech interactions.
"""

import asyncio
import logging
from typing import Callable, Awaitable

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

class VoiceInterface:
    """Voice-based interface for user interaction."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.input_handler: Callable[[str], Awaitable[str]] = None

        if pyttsx3 is None or sr is None:
            missing = []
            if pyttsx3 is None:
                missing.append("pyttsx3")
            if sr is None:
                missing.append("SpeechRecognition")
            raise RuntimeError(
                "VoiceInterface dependencies missing: "
                + ", ".join(missing)
                + ". Install optional voice packages or disable voice interface."
            )

        # TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # STT recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    async def run(self, input_handler: Callable[[str], Awaitable[str]]):
        """Start the voice interface with mode selection."""
        self.input_handler = input_handler
        self.logger.info("Voice interface started.")

        print("AutoBot Voice Mode")
        print("Choose interaction mode:")
        print("1. Text input with voice output")
        print("2. Voice input with voice output")

        while True:
            try:
                mode = input("Enter mode (1 or 2): ").strip()
                if mode == '1':
                    await self.text_with_voice_mode()
                    break
                elif mode == '2':
                    await self.voice_with_voice_mode()
                    break
                else:
                    print("Invalid mode. Please enter 1 or 2.")
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Voice interface error: {e}")
                print("Error in voice interface.")

        self.logger.info("Voice interface stopped.")

    async def text_with_voice_mode(self):
        """Text input, voice output mode."""
        print("Text with Voice mode activated.")
        print("Type your commands. AutoBot will speak responses.")
        print("Type 'quit' to exit, 'switch' to change mode.")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'switch':
                    print("Switching to voice mode...")
                    await self.voice_with_voice_mode()
                    return

                if self.input_handler:
                    response = await self.input_handler(user_input)
                    print(f"AutoBot: {response}")
                    self.speak(response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Text-voice mode error: {e}")
                print("Error processing input.")

    async def voice_with_voice_mode(self):
        """Voice input, voice output mode."""
        print("Voice with Voice mode activated.")
        print("Speak your commands. AutoBot will listen and respond.")
        print("Say 'quit' or 'exit' to stop, 'switch mode' to change.")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Microphone calibrated. Start speaking...")

        while True:
            try:
                with self.microphone as source:
                    print("Listening...")
                    audio = await asyncio.get_event_loop().run_in_executor(None, self.recognizer.listen, source)

                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, self.recognizer.recognize_google, audio
                    )
                    print(f"You said: {user_input}")

                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif 'switch mode' in user_input.lower():
                        print("Switching to text mode...")
                        await self.text_with_voice_mode()
                        return

                    if self.input_handler:
                        response = await self.input_handler(user_input)
                        print(f"AutoBot: {response}")
                        self.speak(response)

                except sr.UnknownValueError:
                    print("Sorry, I didn't understand that.")
                    self.speak("Sorry, I didn't understand that.")
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    self.speak("Speech recognition error.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Voice-voice mode error: {e}")
                print("Error in voice processing.")

    def speak(self, text: str):
        """Convert text to speech."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            print(f"TTS Error: {e}")
