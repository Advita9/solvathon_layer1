# import asyncio
# import threading
# from flask import Flask, request, jsonify
# from aiortc import RTCPeerConnection, RTCSessionDescription
# import numpy as np
# import librosa

# from flask_cors import CORS

# import torch

# from transformers import MimiModel, AutoFeatureExtractor
# from src.models.emergency_classifier import MimiEmergencyClassifier
# from collections import deque
# import time

# from src.models.language_detector import IndicLIDDetector
# lid = IndicLIDDetector()
# current_language = "unknown"

# INPUT_SR = 48000

# EMERGENCY_WINDOW = int(INPUT_SR * 5)   # 0.5 sec
# LANG_WINDOW = int(INPUT_SR * 1.5) 
# DISTRESS_WINDOW = 5        # seconds to track
# MIN_HITS = 3               # how many positives required
# THRESHOLD = 0.7

# distress_events = deque()
# # -----------------------
# # Setup Async Loop in Background
# # -----------------------
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)

# def run_loop():
#     loop.run_forever()

# threading.Thread(target=run_loop, daemon=True).start()

# # -----------------------
# # Flask App
# # -----------------------
# app = Flask(__name__)

# CORS(app) 
# pcs = set()

# device = "mps" if torch.backends.mps.is_available() else "cpu"

# print("Loading Mimi...")
# mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
# fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
# TARGET_SR = fe.sampling_rate

# print("Loading classifier...")
# clf = MimiEmergencyClassifier().to(device)
# clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
# clf.eval()


# # -----------------------
# # Audio Processing
# # -----------------------
# async def process_audio_track(track):
#     global current_language

#     print("🎙 Audio track started")

#     buffer = np.array([], dtype=np.float32)
#     last_lang_detect_samples = 0

#     while True:
#         frame = await track.recv()
#         pcm = frame.to_ndarray()

#         if pcm.ndim > 1:
#             pcm = pcm.mean(axis=0)

#         pcm = pcm.astype(np.float32) / 32768.0
#         buffer = np.concatenate([buffer, pcm])

#         # print("Frame shape:", pcm.shape, "Buffer length:", len(buffer))

#         # 🌍 Language Detection (every 1.5 sec)
#         if len(buffer) - last_lang_detect_samples >= LANG_WINDOW:
#             segment = buffer[last_lang_detect_samples:last_lang_detect_samples + LANG_WINDOW]

#             try:
#                 lang, conf = lid.detect_from_audio(segment, sr=INPUT_SR)
#                 current_language = lang
#                 print(f"🌍 Detected language: {lang} ({conf:.2f})")
#             except Exception as e:
#                 print("LID error:", e)

#             last_lang_detect_samples = len(buffer)

#         # 🚑 Emergency Detection (every 0.5 sec)
#         if len(buffer) >= EMERGENCY_WINDOW:
#             chunk = buffer[:EMERGENCY_WINDOW]
#             buffer = buffer[EMERGENCY_WINDOW:]

#             await run_emergency_detection(chunk)


# async def run_emergency_detection(audio_chunk):
#     global distress_events

#     audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)
#     inputs = fe(raw_audio=audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

#     with torch.no_grad():
#         codes = mimi.encode(inputs["input_values"]).audio_codes
#         prob = clf(codes).item()

#     print(f"🚑 Emergency probability: {prob:.3f}")

#     now = time.time()

#     # Remove old events outside window
#     while distress_events and now - distress_events[0] > DISTRESS_WINDOW:
#         distress_events.popleft()

#     # Add new event if above threshold
#     if prob > THRESHOLD:
#         distress_events.append(now)

#     # Check sustained distress
#     if len(distress_events) >= MIN_HITS:
#         print("🚨 SUSTAINED DISTRESS DETECTED 🚨")
#         distress_events.clear()  # prevent repeat spam



# # -----------------------
# # WebRTC Offer Handling
# # -----------------------
# @app.route("/offer", methods=["POST"])
# def offer():
#     params = request.json
#     offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

#     pc = RTCPeerConnection()
#     pcs.add(pc)

#     @pc.on("track")
#     def on_track(track):
#         if track.kind == "audio":
#             print("🎧 Receiving audio from browser")
#             asyncio.run_coroutine_threadsafe(process_audio_track(track), loop)

#     async def handle_offer():
#         await pc.setRemoteDescription(offer)
#         answer = await pc.createAnswer()
#         await pc.setLocalDescription(answer)
#         return pc.localDescription

#     future = asyncio.run_coroutine_threadsafe(handle_offer(), loop)
#     local_desc = future.result()

#     return jsonify({"sdp": local_desc.sdp, "type": local_desc.type})


# # -----------------------
# # Start Server
# # -----------------------
# if __name__ == "__main__":
#     print("🚀 Signaling + Audio Processing Server Running")
#     app.run("0.0.0.0", 8080)



import asyncio
import threading
from flask import Flask, request, jsonify
from aiortc import RTCPeerConnection, RTCSessionDescription
import numpy as np
import librosa
import os  # Restored missing import

from src.llm.embedding_agent import process_with_context, context_manager as llm_context
from src.llm.stream_chunker import StreamChunker
from src.tts.tts_manager import TTSManager
from src.realtime.audio_track import TTSAudioTrack
from src.realtime.twilio_track import TwilioInputTrack
from flask_sock import Sock
import audioop
import base64
import json

tts_manager = TTSManager()

import torch

from transformers import MimiModel, AutoFeatureExtractor
from src.models.emergency_classifier import MimiEmergencyClassifier
from collections import deque
import time

DISTRESS_WINDOW = 5        # seconds to track
MIN_HITS = 3               # how many positives required
THRESHOLD = 0.7
# -----------------------
# Whisper Endpointing State
# -----------------------
asr_buffer = np.array([], dtype=np.float32)

SILENCE_THRESHOLD = 0.005   # speech energy level
SILENCE_DURATION = 2.0      # seconds of silence = end of utterance
MIN_SPEECH_DURATION = 1.0   # ignore tiny noises

last_voice_time = None
speech_start_time = None

distress_events = deque()
# -----------------------
# Setup Async Loop in Background
# -----------------------
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_loop():
    loop.run_forever()

threading.Thread(target=run_loop, daemon=True).start()

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)
sock = Sock(app)

CORS(app) 
pcs = set()

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading Mimi...")
mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
TARGET_SR = fe.sampling_rate

print("Loading classifier...")
clf = MimiEmergencyClassifier().to(device)
clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
clf.eval()



from faster_whisper import WhisperModel
import threading

whisper_model = None

def load_whisper():
    global whisper_model
    print("🔄 Loading Whisper model in background...")

    whisper_model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"   # ← FIXED
    )

    print("✅ Whisper ASR ready!")


threading.Thread(target=load_whisper, daemon=True).start()

# -----------------------
# MMS-LID Language Detection (uses raw audio - no extra normalization)
# -----------------------
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor as LIDFeatureExtractor

INPUT_SR = 48000
LID_TARGET_SR = 16000  # MMS-LID expects 16kHz
LANG_WINDOW = int(INPUT_SR * 1.5)  # 1.5 sec of audio for language detection

current_language = "unknown"

# Language code mappings
LANGUAGE_CODE_MAP = {
    'hin': 'hi', 'tam': 'ta', 'tel': 'te', 'kan': 'kn',
    'ben': 'bn', 'guj': 'gu', 'mal': 'ml', 'mar': 'mr',
    'pan': 'pa', 'ori': 'or', 'asm': 'as', 'urd': 'ur',
    'eng': 'en',
}

LANGUAGE_NAMES = {
    'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'kn': 'Kannada',
    'bn': 'Bengali', 'gu': 'Gujarati', 'ml': 'Malayalam', 'mr': 'Marathi',
    'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'ur': 'Urdu',
}

INDIAN_LANGUAGES = {'hin', 'tam', 'tel', 'kan', 'ben', 'guj', 'mal', 'mar', 'pan', 'ori', 'asm', 'urd'}

lid_model = None
lid_processor = None
lid_id2label = None

def load_lid_model():
    global lid_model, lid_processor, lid_id2label
    print("🔄 Loading MMS-LID language detector in background...")
    
    lid_processor = LIDFeatureExtractor.from_pretrained("facebook/mms-lid-256")
    lid_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/mms-lid-256")
    lid_model = lid_model.to(device)
    lid_model.eval()
    lid_id2label = lid_model.config.id2label
    
    print(f"✅ MMS-LID ready! ({len(lid_id2label)} languages)")

threading.Thread(target=load_lid_model, daemon=True).start()

# -----------------------
# Audio Processing
# -----------------------
import io # Ensure io is imported

async def process_audio_track(track, target_language="eng", output_track=None, twilio_output_sender=None):
    print(f"🎙 Audio track started (Language: {target_language})")

    # Create LLM Session for this track
    try:
        session_id = llm_context.create_session(language=target_language)
        print(f"🆕 Started LLM Session: {session_id}")
    except Exception as e:
        print(f"⚠️ Failed to create Redis session: {e}")
        session_id = "temp_session"


    # Create local buffer for this track session (don't use global asr_buffer safely multiple calls)
    # Actually, using global is bad for multiple calls. Let's make it local.
    local_asr_buffer = np.array([], dtype=np.float32)
    local_speech_start = None
    local_last_voice = None

    buffer = np.array([], dtype=np.float32)

    # Tuned for browser mic
    SILENCE_THRESHOLD = 0.01      # higher = less noise triggering
    SILENCE_DURATION = 2.0        # seconds
    MIN_SPEECH_DURATION = 1.0

    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)

        pcm = pcm.astype(np.float32) / 32768.0

        # ---------------- EMERGENCY DETECTION ----------------
        buffer = np.concatenate([buffer, pcm])

        if len(buffer) >= TARGET_SR:
            chunk = buffer[:TARGET_SR]
            buffer = buffer[TARGET_SR:]
            await run_emergency_detection(chunk)

        # ---------------- WHISPER ENDPOINTING ----------------
        energy = np.mean(np.abs(pcm))
        now = time.time()

        if energy > SILENCE_THRESHOLD:
            # Speech detected
            if local_speech_start is None:
                local_speech_start = now
                print("🗣 Speech started")

            local_last_voice = now
            local_asr_buffer = np.concatenate([local_asr_buffer, pcm])

        else:
            # Silence frame
            if local_speech_start is not None and local_last_voice is not None:
                silence_time = now - local_last_voice

                if silence_time > SILENCE_DURATION:
                    speech_length = local_last_voice - local_speech_start

                    if speech_length > MIN_SPEECH_DURATION and len(local_asr_buffer) > 0:
                        print(f"🛑 Speech ended → Sending to Whisper ({target_language})")
                        chunk = local_asr_buffer.copy()
                        asyncio.create_task(run_whisper_asr(
                            chunk, 
                            language_code=target_language, 
                            session_id=session_id,
                            output_track=output_track,
                            twilio_output_sender=twilio_output_sender
                        ))

                    # Reset state
                    local_asr_buffer = np.array([], dtype=np.float32)
                    local_speech_start = None
                    local_last_voice = None


# async def run_whisper_asr(audio_chunk_48k):
#     global whisper_model

#     if whisper_model is None:
#         return

#     print("🧠 Running Whisper ASR...")

#     audio_16k = librosa.resample(audio_chunk_48k, orig_sr=48000, target_sr=16000)
#     audio_16k = audio_16k.astype(np.float32)

#     # Boost mic level a bit
#     audio_16k *= 2.5
#     audio_16k = np.clip(audio_16k, -1.0, 1.0)

#     segments, info = whisper_model.transcribe(
#         audio_16k,
#         beam_size=1,
#         vad_filter=True,
#         vad_parameters=dict(min_silence_duration_ms=300)
#     )

#     text = " ".join(seg.text for seg in segments).strip()

#     if text:
#         print(f"📝 ASR: {text}")
#     else:
#         print("🤐 (No speech detected)")

async def run_whisper_asr(audio_chunk_48k, language_code="eng", session_id="temp_session", output_track=None, twilio_output_sender=None):
    global whisper_model

    if whisper_model is None:
        print("⏳ Whisper still loading...")
        return

    whisper_lang = LANGUAGE_CODE_MAP.get(language_code, 'en')
    print(f"🧠 Running Whisper ASR on full utterance (Lang: {language_code} -> {whisper_lang})...")

    audio_16k = librosa.resample(audio_chunk_48k, orig_sr=48000, target_sr=16000)
    audio_np = audio_16k.astype(np.float32)

    audio_np *= 2.0
    audio_np = np.clip(audio_np, -1.0, 1.0)

    segments, info = whisper_model.transcribe(
        audio_np,
        beam_size=1,
        language=whisper_lang,
        vad_filter=False
    )

    text = " ".join(seg.text for seg in segments).strip()

    if text:
        print(f"📝 ASR Final: {text}")

        # --- LLM INTEGRATION (STREAMING) ---
        print(f"🤖 Sending to LLM (Session: {session_id})...")
        
        # Metadata Setup
        sustained_distress = len(distress_events) >= MIN_HITS
        metadata = {
            "detected_language": language_code, 
            "language_confidence": "0.99",
            "emergency_probability": "High" if sustained_distress else "Low",
            "sustained_distress": str(sustained_distress),
            "processing_latency_ms": "100"
        }

        # Define the streaming worker (Runs in Thread)
        def stream_worker():
            try:
                # 1. Start Stream
                llm_result = process_with_context(
                    prompt=text, 
                    session_id=session_id,
                    similarity_threshold=0.85,
                    metadata=metadata,
                    stream=True  # Enable Streaming
                )
                
                # Check if it returned a stream or cached response
                if not llm_result.get("stream"):
                    # Cached/Static response handling
                    response_text = llm_result["response"]
                    print(f"✅ [Cache] Response: {response_text}")
                    # Send full text to TTS
                    asyncio.run_coroutine_threadsafe(
                        tts_manager.speak(response_text, language_code), loop
                    )
                    return

                # 2. Process Stream
                print("🌊 LLM Stream Started...")
                chunker = StreamChunker()
                chunk_count = 0
                
                # Pass filler (if any) first
                filler = llm_result.get("filler_audio")
                if filler:
                    print(f"🤔 Filler Triggered")
                    # TODO: Send filler audio bytes directly
                
                # Define a helper to run TTS and handle output
                async def run_tts_and_send(text_chunk, lang):
                     audio_bytes = await tts_manager.speak(text_chunk, lang)
                     if audio_bytes:
                         print(f"🔊 Generated {len(audio_bytes)} bytes of audio for: '{text_chunk[:10]}...'")
                         
                         if output_track:
                             try:
                                 # Decode and queue logic is sync but inside async wrapper helper if needed
                                 # Track's add_audio_bytes is synchronous (cpu bound decode)
                                 # Better to run in executor if heavy, but for small chunks it's okay-ish 
                                 # or wrap in run_in_executor
                                 loop = asyncio.get_running_loop()
                                 await loop.run_in_executor(None, output_track.add_audio_bytes, audio_bytes)
                             except Exception as e:
                                 print(f"❌ Failed to add audio to track: {e}")
                                 
                         if twilio_output_sender:
                             # Send to Phone
                             await twilio_output_sender(audio_bytes)
                                  
                         # Save to debug (optional)
                         # with open(f"debug_tts_{int(time.time())}.wav", "wb") as f:
                         #    f.write(audio_bytes)
                     else:
                         print("⚠️ TTS Synthesis Failed")

                generator = llm_result["response_generator"]
                
                for token in generator:
                    # Feed to chunker
                    for chunk in chunker.process(token):
                        chunk_count += 1
                        print(f"🗣️  [Chunk #{chunk_count}] {chunk}")
                        
                        # Schedule TTS task on main loop (with output handling)
                        asyncio.run_coroutine_threadsafe(
                            run_tts_and_send(chunk, language_code), loop
                        )
                
                # Flush remaining
                last_chunk = chunker.flush()
                if last_chunk:
                    chunk_count += 1
                    print(f"🗣️  [Chunk #{chunk_count}] {last_chunk}")
                    asyncio.run_coroutine_threadsafe(
                        run_tts_and_send(last_chunk, language_code), loop
                    )
                    
                print(f"✅ Stream Complete ({chunk_count} chunks)")
                
            except Exception as e:
                print(f"❌ LLM Stream Error: {e}")

        # Start the worker thread
        threading.Thread(target=stream_worker, daemon=True).start()

    else:
        print("🤐 No speech detected")

async def run_emergency_detection(audio_chunk):
    global distress_events

    audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)
    inputs = fe(raw_audio=audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"]).audio_codes
        prob = clf(codes).item()

    print(f"🚑 Emergency probability: {prob:.3f}")

    now = time.time()

    # Remove old events outside window
    while distress_events and now - distress_events[0] > DISTRESS_WINDOW:
        distress_events.popleft()

    # Add new event if above threshold
    if prob > THRESHOLD:
        distress_events.append(now)

    # Check sustained distress
    if len(distress_events) >= MIN_HITS:
        print("🚨 SUSTAINED DISTRESS DETECTED 🚨")
        distress_events.clear()  # prevent repeat spam



# -----------------------
# WebRTC Offer Handling
# -----------------------
@app.route("/offer", methods=["POST"])
def offer():
    params = request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    client_language = params.get("language", "eng")

    pc = RTCPeerConnection()
    pcs.add(pc)
    
    # Create Speaker Track
    output_track = TTSAudioTrack()
    pc.addTrack(output_track)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print(f"🎧 Receiving audio from browser (Lang: {client_language})")
            asyncio.run_coroutine_threadsafe(
                process_audio_track(track, client_language, output_track=output_track), 
                loop
            )

    async def handle_offer():
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    future = asyncio.run_coroutine_threadsafe(handle_offer(), loop)
    local_desc = future.result()

    return jsonify({"sdp": local_desc.sdp, "type": local_desc.type})


# -----------------------
# Start Server
# -----------------------
# -----------------------
# Start Server
# -----------------------
@app.route("/lid-test")
def lid_test_page():
    return open("src/realtime/lid_test.html").read()

@app.route("/detect-lang", methods=["POST"])
def detect_lang_endpoint():
    global lid_model, lid_processor, lid_id2label

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    file = request.files["audio"]
    temp_path = f"lid_debug/upload_{int(time.time())}.wav"
    os.makedirs("lid_debug", exist_ok=True)
    file.save(temp_path)

    try:
        # Load exactly like Colab
        waveform, sr = librosa.load(temp_path, sr=16000, mono=True)
        
        # Energy check
        energy = np.mean(np.abs(waveform))
        
        # Inference
        inputs = lid_processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = lid_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        ranked = sorted(
            [(lid_id2label[i], probs[i].item()) for i in range(len(probs))],
            key=lambda x: x[1],
            reverse=True
        )

        # Build top lists
        top_global_list = [{"code": r[0], "prob": r[1]} for r in ranked[:3]]

        # Filter for Indian
        indian_results = []
        for lang, p in ranked:
            if lang in INDIAN_LANGUAGES:
                indian_results.append({"code": lang, "prob": p, "name": LANGUAGE_NAMES.get(LANGUAGE_CODE_MAP.get(lang, lang), lang)})

        # Choose highest-confidence overall language (top_global[0]) if available, else fallback to top_indian
        selected_lang = None
        selected_prob = None
        if top_global_list:
            selected_lang = top_global_list[0]["code"]
            selected_prob = top_global_list[0]["prob"]
        elif indian_results:
            selected_lang = indian_results[0]["code"]
            selected_prob = indian_results[0]["prob"]

        # Persist selection to server state for this process
        try:
            if selected_lang is not None:
                # set current_language to selected (store raw model code)
                current_language = selected_lang
        except Exception:
            # ignore if current_language not used elsewhere
            pass

        # Log the full result prefixed with Done! and a concise line with chosen language
        result_payload = {
            "status": "success",
            "energy": float(energy),
            "top_indian": indian_results[:3],
            "top_global": top_global_list,
            "duration": len(waveform)/16000
        }

        print("Done!", result_payload)
        if selected_lang is not None:
            print(f"🌍 Language detected (highest global confidence): {selected_lang} (prob={selected_prob:.4f})")

        return jsonify(result_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Signaling + Audio Processing Server Running")
    app.run("0.0.0.0", 8080)