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

from flask_cors import CORS

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
# Audio Processing
# -----------------------
async def process_audio_track(track):
    print("🎙 Audio track started")

    global asr_buffer, last_voice_time, speech_start_time

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
            if speech_start_time is None:
                speech_start_time = now
                print("🗣 Speech started")

            last_voice_time = now
            asr_buffer = np.concatenate([asr_buffer, pcm])

        else:
            # Silence frame
            if speech_start_time is not None and last_voice_time is not None:
                silence_time = now - last_voice_time

                if silence_time > SILENCE_DURATION:
                    speech_length = last_voice_time - speech_start_time

                    if speech_length > MIN_SPEECH_DURATION and len(asr_buffer) > 0:
                        print("🛑 Speech ended → Sending to Whisper")
                        chunk = asr_buffer.copy()
                        asyncio.create_task(run_whisper_asr(chunk))

                    # Reset state
                    asr_buffer = np.array([], dtype=np.float32)
                    speech_start_time = None
                    last_voice_time = None


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

async def run_whisper_asr(audio_chunk_48k):
    global whisper_model

    if whisper_model is None:
        print("⏳ Whisper still loading...")
        return

    print("🧠 Running Whisper ASR on full utterance...")

    audio_16k = librosa.resample(audio_chunk_48k, orig_sr=48000, target_sr=16000)
    audio_np = audio_16k.astype(np.float32)

    audio_np *= 2.0
    audio_np = np.clip(audio_np, -1.0, 1.0)

    segments, info = whisper_model.transcribe(
        audio_np,
        beam_size=1,
        vad_filter=False
    )

    text = " ".join(seg.text for seg in segments).strip()

    if text:
        print(f"📝 ASR Final: {text}")
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

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("🎧 Receiving audio from browser")
            asyncio.run_coroutine_threadsafe(process_audio_track(track), loop)

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
if __name__ == "__main__":
    print("🚀 Signaling + Audio Processing Server Running")
    app.run("0.0.0.0", 8080)