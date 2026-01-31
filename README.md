# Solvathon Layer 1: Real-Time Voice AI

> **Low-latency, multilingual, empathetic voice intelligence for emergency response.**

This repository implements the **first layer** of a multi-tiered emergency response AI. It handles the immediate vocal interaction with a caller, determining their language, emotional state, and intent in real-time.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** running locally (`llama3.2:3b` model pulled).
- **Redis** running locally.

### 2. Installation
```bash
# Clone the repo
git clone <repo-url>
cd solvathon_layer1

# Install Dependencies
pip install -r requirements.txt
python3 -m pip install edge-tts piper-tts

# Download TTS Models
chmod +x src/tts/setup_piper.sh
./src/tts/setup_piper.sh
```

### 3. Run
```bash
python src/realtime/signaling_server.py
```
Open [http://localhost:8080](http://localhost:8080) in your browser.

---

## 📚 Documentation
For detailed system architecture, configuration, and API reference, please see the **[Technical Manual](./TECHNICAL_MANUAL.md)**.

## ✨ Key Features
- **🗣️ Multilingual**: Fluent in English, Hindi, Tamil, Telugu, Kannada.
- **⚡ Real-time**: <500ms latency using streamed LLM tokens.
- **🚑 Emergency Aware**: Detects screaming/distress and switches to "Empathy Mode".
- **🔊 Hybrid TTS**: Combines offline speed (Piper) with online quality (Edge TTS).
