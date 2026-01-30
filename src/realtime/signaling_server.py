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


# -----------------------
# Audio Processing
# -----------------------
async def process_audio_track(track):
    print("🎙 Audio track started")
    buffer = np.array([], dtype=np.float32)

    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)

        pcm = pcm.astype(np.float32) / 32768.0
        buffer = np.concatenate([buffer, pcm])

        if len(buffer) >= TARGET_SR:
            chunk = buffer[:TARGET_SR]
            buffer = buffer[TARGET_SR:]

            await run_emergency_detection(chunk)


async def run_emergency_detection(audio_chunk):
    audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)

    inputs = fe(raw_audio=audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"]).audio_codes
        prob = clf(codes).item()

    print(f"🚑 Emergency probability: {prob:.3f}")

    if prob > 0.7:
        print("🚨 DISTRESS DETECTED 🚨")


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
