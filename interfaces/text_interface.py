"""
Text Interface for AutoBot
Handles terminal-based text input and output.
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False

class TextInterface:
    """Simple text-based interface for user interaction."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.input_handler: Optional[Callable[[str], Awaitable[str]]] = None

        # Initialize TTS engine
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 180)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            self.logger.info("TTS engine initialized")
        else:
            self.tts_engine = None
            self.logger.warning("TTS not available - pyttsx3 not installed")

    async def run(self, input_handler: Callable[[str], Awaitable[str]]):
        """Start the text interface loop."""
        self.input_handler = input_handler
        self.logger.info("Text interface started.")

        print(f"{self.config['assistant']['name']} v{self.config['assistant']['version']} ready.")
        print("Type your commands or 'quit' to exit.")
        if self.tts_engine:
            print("Text-to-speech is enabled - AutoBot will speak responses!")

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                if user_input.lower() in ['quit', 'exit']:
                    break

                # Skip empty inputs
                if not user_input or not user_input.strip():
                    continue

                if self.input_handler:
                    response = await self.input_handler(user_input)
                    print(f"AutoBot: {response}")

                    # Speak the response (fire-and-forget, don't await)
                    if self.tts_engine and response:
                        asyncio.create_task(self._speak_response(response))

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Text interface error: {e}")
                print("Error processing input.")

        self.logger.info("Text interface stopped.")

    async def _speak_response(self, response: str):
        """Speak the response using TTS."""
        try:
            # Clean response for speech (remove any special characters that might cause issues)
            clean_response = response.replace('\n', ' ').strip()
            if clean_response:
                # Start TTS in background without blocking
                def speak():
                    try:
                        # Create a new TTS engine instance for this thread to avoid conflicts
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 180)
                        engine.setProperty('volume', 0.9)
                        engine.say(clean_response)
                        engine.runAndWait()
                        engine.stop()
                    except Exception as e:
                        print(f"TTS execution error: {e}")

                # Run TTS in a separate thread without awaiting
                import threading
                thread = threading.Thread(target=speak, daemon=True)
                thread.start()

        except Exception as e:
            self.logger.error(f"TTS setup error: {e}")