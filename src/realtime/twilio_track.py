import asyncio
import io
import av
import audioop
import base64
from aiortc import MediaStreamTrack
import time
import numpy as np

class TwilioInputTrack(MediaStreamTrack):
    """
    Receives u-law 8000Hz audio from Twilio and converts it to PCM 48000Hz for the AI pipeline.
    """
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.samplerate = 48000
        self.channels = 1 
        self._start_time = None
        self._timestamp = 0

    def add_mulaw_chunk(self, mulaw_payload: str):
        """
        Takes a base64 encoded mulaw chunk from Twilio.
        """
        try:
            # 1. Decode Base64
            mulaw_bytes = base64.b64decode(mulaw_payload)
            
            # 2. Convert mulaw (8kHz) -> PCM 16-bit (8kHz)
            # audioop.ulaw2lin(fragment, width) -> width=2 for 16-bit
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
            
            # 3. Resample 8kHz -> 48kHz
            # Using audioop.ratecv(fragment, width, nchannels, inrate, outrate, state)
            # state can be None
            pcm_48k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 48000, None)
            
            # 4. Create AudioFrame
            frame = av.AudioFrame(format='s16', layout='mono', samples=len(pcm_48k)//2)
            frame.planes[0].update(pcm_48k)
            frame.sample_rate = 48000
            frame.time_base = av.AudioFrame(0).time_base # Placeholder
            
            self.queue.put_nowait(frame)
            
        except Exception as e:
            print(f"❌ Twilio Decode Error: {e}")

    async def recv(self):
        if self._start_time is None:
            self._start_time = time.time()

        frame = await self.queue.get()
        
        # Timing logic
        frame.pts = self._timestamp
        self._timestamp += frame.samples
        
        return frame
