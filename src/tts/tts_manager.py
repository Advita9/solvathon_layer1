import asyncio
import os
from typing import Optional

class TTSManager:
    """
    Manages Text-to-Speech synthesis, routing requests to the appropriate engine
    based on language and latency requirements.
    
    Engines:
    - Piper (Local): English, Hindi, Telugu (Low latency)
    - Edge TTS (Cloud): Tamil, Kannada (High quality)
    """
    
    def __init__(self):
        # Path to Piper binary (from pip install or manual)
        self.piper_path = "/Users/jeevithg/Library/Python/3.9/bin/piper"
        if not os.path.exists(self.piper_path):
             # Fallback to just 'piper' if in PATH
             self.piper_path = "piper"
             
        self.models_dir = os.path.abspath("src/tts/piper_models")
        self.output_dir = "tts_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_engine(self, language_code: str) -> str:
        if language_code in ["en", "hi", "te"]:
            return "piper"
        return "edge"

    async def speak(self, text: str, language_code: str) -> Optional[bytes]:
        """
        Synthesize speech from text.
        """
        # engine = self.get_engine(language_code) # DISABLED LOG to reduce spam
        # print(f"🔊 [TTS] Synthesizing with {engine.upper()}: '{text[:20]}...'")
        
        if self.get_engine(language_code) == "piper":
            return await self._speak_piper(text, language_code)
        else:
            return await self._speak_edge(text, language_code)
            
    async def _speak_piper(self, text: str, language_code: str) -> Optional[bytes]:
        """
        Generate audio using Piper TTS (Local).
        """
        # Select Model
        model_name = "en_US-lessac-medium.onnx"
        if language_code == "hi":
            model_name = "hi_IN-dhananjai-x_low.onnx" # Fallback to x_low
        elif language_code == "te":
            model_name = "te_IN-maya-medium.onnx"
            
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"⚠️ Piper model not found: {model_path}. Fallback to Edge TTS.")
            return await self._speak_edge(text, language_code)

        try:
            # Run Piper
            proc = await asyncio.create_subprocess_exec(
                self.piper_path,
                "--model", model_path,
                "--output-raw", # Output raw PCM
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate(input=text.encode())
            
            if proc.returncode != 0:
                print(f"❌ Piper Error: {stderr.decode()}")
                return None
                
            # Convert raw PCM to WAV (optional, but easier for browser)
            # Actually, standard Piper output with --output-raw is PCM. 
            # We want WAV for easy playing? 
            # If we omit --output-raw, it outputs WAV to stdout.
            # Let's use clean WAV output.
            
            # Re-run without --output-raw to get WAV container
            proc = await asyncio.create_subprocess_exec(
                self.piper_path,
                "--model", model_path,
                "--output_file", "-", # stdout
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate(input=text.encode())
             
            if proc.returncode != 0:
                 print(f"❌ Piper Error: {stderr.decode()}")
                 return None
                 
            return stdout
            
        except Exception as e:
            print(f"❌ Piper Exception: {e}")
            return None

    async def _speak_edge(self, text: str, language_code: str) -> Optional[bytes]:
        """
        Generate audio using Edge TTS (Cloud).
        """
        import edge_tts
        
        # Select Voice
        voice = "ta-IN-PallaviNeural" # Tamil Default
        if language_code == "kn":
            voice = "kn-IN-GaganNeural"
        elif language_code == "te":
            voice = "te-IN-MohanNeural" # Fallback for Telugu if Piper fails
        elif language_code == "hi":
            voice = "hi-IN-SwaraNeural"
        
        try:
            communicate = edge_tts.Communicate(text, voice)
            # We can't easily stream bytes from edge_tts in one go without a file strictly? 
            # Actually it has a stream() method.
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Edge TTS Exception: {e}")
            return None
