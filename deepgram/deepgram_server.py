import asyncio
import json
import os
import logging
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.info import Info, Describe, AsrProgram, AsrModel, Attribution

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPTIONS_FILE = "/data/options.json"

class DeepgramSTT:
    def __init__(self):
        deepgram_api_key = load_api_key()
        self.dg_client = DeepgramClient(deepgram_api_key)

    async def transcribe(self, audio_data: bytes, sample_rate: int):
        """
        Send audio data to Deepgram and return transcription.
        """
        options: PrerecordedOptions = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            encoding='linear16',
            sample_rate=sample_rate,
            channels=1,
            language='en-US',
            punctuate=True,
        )
        payload: FileSource = {"buffer": audio_data, "mimetype": "audio/wav"}
        
        response = self.dg_client.listen.rest.v("1").transcribe_file(payload, options)

        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        return transcript

class State:
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def set_session(self, session_id, data):
        self.sessions[session_id] = data

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

class EventHandler(AsyncEventHandler):
    WYOMING_INFO = Info(
        asr=[
            AsrProgram(
                name="Deepgram",
                description="A speech recognition toolkit",
                attribution=Attribution(
                    name="Deepgram",
                    url="https://deepgram.com",
                ),
                installed=True,
                version='1.0',
                models=[
                    AsrModel(
                        name='general-nova-3',
                        description='Nova 3',
                        attribution=Attribution(
                            name="Deepgram",
                            url="https://deepgram.com",
                        ),
                        installed=True,
                        version=None,
                        languages=['en'],
                    )
                ]
            )
        ]
    )

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initialize Wyoming event handler."""
        logger.info(f"✅ New connection")

        super().__init__(*args, **kwargs)

        state = State()
        self.stt = DeepgramSTT()
        self.audio_data = b""
        self.sample_rate = 16000  # Default sample rate; can be adjusted
        wyoming_info = self.WYOMING_INFO
        self.wyoming_info_event = wyoming_info.event()

    async def handle_event(self, event: Event) -> bool:
        """Process and log all incoming Wyoming protocol events."""
        if event.type == "describe":
            logger.info("📤 Responding to describe event.")
            await self.write_event(self.wyoming_info_event)
        elif event.type == "audio-chunk":
            self.audio_data += event.payload
        elif event.type == "audio-stop":
            logger.info(f"Received Wyoming event: {event.type} - Data: {event.data}")
            # Send to Deepgram and get transcription
            text = await self.stt.transcribe(self.audio_data, self.sample_rate)
            result_event = Event(type="transcript", data={"text": text})
            
            logger.info(f"Sending Transcript Event: {text}")
            
            await self.write_event(result_event)
            self.audio_data = b""  # Reset for next transcription
        else:
            logger.info(f"Received Wyoming event: {event.type} - Data: {event.data}")

        return True

def load_api_key():
    """Load the API key from the options.json file."""
    try:
        with open(OPTIONS_FILE, "r") as f:
            options = json.load(f)
        api_key = options.get("api_key", "")
        logger.info(f"✅ API Key Loaded: {api_key[:4]}****")
        return api_key
    except Exception as e:
        logger.error(f"Error reading options.json: {e}")
        logger.error("⚠️ API key is missing! Set it in the Deepgram addon settings.")
        return ""

async def main():
    """Starts the Wyoming Deepgram STT server using DeepgramServer."""
    server = AsyncServer.from_uri('tcp://0.0.0.0:10301')
    try:
        logger.info('Starting Wyoming Server')
        await server.run(EventHandler)
    except asyncio.CancelledError:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())
