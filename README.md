# AI Translator — FastAPI Backend v2.0

Full multilingual pipeline for Indian languages.  
**Input:** Text · OCR (image) · Speech (audio)  
**Output:** Translated text + synthesised speech (WAV / base64)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  /text/      │  │  /ocr/       │  │  /speech/            │   │
│  │  translate   │  │  extract     │  │  transcribe-and-     │   │
│  │              │  │  extract-and │  │  translate           │   │
│  │              │  │  -translate  │  │                      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
│         │                 │                      │               │
│         ▼                 ▼                      ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ServiceRegistry                        │   │
│  │              (Lazy loading · Singleton)                   │   │
│  └──────┬──────────────┬──────────────┬──────────┬──────────┘   │
│         │              │              │          │               │
│         ▼              ▼              ▼          ▼               │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────┐  │
│  │Translation │ │    OCR     │ │   STT    │ │     TTS        │  │
│  │ Service    │ │  Service   │ │ Service  │ │   Service      │  │
│  │            │ │            │ │          │ │                │  │
│  │ IndicTrans2│ │ PaddleOCR  │ │ Whisper  │ │ Indic Parler   │  │
│  │ (primary)  │ │ (primary)  │ │ (base)   │ │ (primary)      │  │
│  │            │ │ Tesseract  │ │          │ │                │  │
│  │ M2M-100    │ │ (fallback) │ │          │ │ eSpeak-NG      │  │
│  │ (commented │ │ Qwen LLM   │ │          │ │ (fallback)     │  │
│  │  fallback) │ │ (correction│ │          │ │                │  │
│  └────────────┘ └────────────┘ └──────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Lazy + Dynamic Loading

All models are loaded **on first use**, not at startup:
- Server starts in ~1 second
- First request to each pipeline loads that pipeline's models
- Models stay in memory (warm cache) for subsequent requests
- Services load in parallel where possible (asyncio tasks)

---

## Setup

### 1. System dependencies

```bash
# OCR
sudo apt install tesseract-ocr \
  tesseract-ocr-hin tesseract-ocr-tam tesseract-ocr-tel \
  tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-ben \
  tesseract-ocr-guj tesseract-ocr-mar tesseract-ocr-pan

# TTS fallback
sudo apt install espeak-ng -y
```

### 2. Python packages

```bash
pip install -r requirements.txt

# IndicTransToolkit (required for translation)
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit && pip install --editable . --use-pep517
cd ..

# Indic Parler TTS
pip install git+https://github.com/huggingface/parler-tts
```

### 3. LLM (optional — for OCR correction + TTS enhancement)

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen2.5:1.5b
```

### 4. Run

```bash
cd ai_translator_api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

---

## API Reference

### Text Pipeline

#### `POST /api/v1/text/translate`

```json
{
  "text": "Hello, how are you?",
  "target_lang": "tam_Taml",
  "source_lang": null,
  "tts": {
    "enabled": true,
    "engine": "auto",
    "return_audio": true,
    "speed": 150
  }
}
```

Response:
```json
{
  "success": true,
  "input_text": "Hello, how are you?",
  "detected_source_lang": "eng_Latn",
  "detected_source_lang_name": "English",
  "target_lang": "tam_Taml",
  "target_lang_name": "Tamil",
  "translated_text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
  "translation_engine": "indictrans2",
  "tts": {
    "success": true,
    "engine_used": "indic_parler",
    "audio_base64": "UklGRi...",
    "audio_format": "wav",
    "sample_rate": 22050,
    "processing_ms": 1240
  },
  "processing_ms": 2100
}
```

#### `POST /api/v1/text/detect-language`

```json
{ "text": "நான் தமிழ் பேசுகிறேன்" }
```

---

### OCR Pipeline

#### `POST /api/v1/ocr/extract`

```
Content-Type: multipart/form-data

file: <image file>
source_lang_hint: tam_Taml  (optional)
```

#### `POST /api/v1/ocr/extract-and-translate`

```
Content-Type: multipart/form-data

file: <image file>
target_lang: eng_Latn
source_lang_hint: tam_Taml  (optional)
tts_enabled: true
tts_return_audio: true
```

---

### Speech Pipeline

#### `POST /api/v1/speech/transcribe`

```
Content-Type: multipart/form-data

file: <audio file>  (WAV / MP3 / OGG / FLAC)
```

#### `POST /api/v1/speech/transcribe-and-translate`

```
Content-Type: multipart/form-data

file: <audio file>
target_lang: eng_Latn
tts_enabled: true
tts_return_audio: true
```

---

### Health

| Endpoint | Description |
|---|---|
| `GET /health` | Quick liveness check |
| `GET /health/status` | Which models are loaded |
| `GET /health/languages` | All supported language codes |

---

## Supported Languages

| Code | Language |
|---|---|
| `eng_Latn` | English |
| `tam_Taml` | Tamil |
| `hin_Deva` | Hindi |
| `tel_Telu` | Telugu |
| `kan_Knda` | Kannada |
| `mal_Mlym` | Malayalam |
| `ben_Beng` | Bengali |
| `guj_Gujr` | Gujarati |
| `mar_Deva` | Marathi |
| `pan_Guru` | Punjabi |
| `ory_Orya` | Odia |
| `urd_Arab` | Urdu |
| `asm_Beng` | Assamese |

---

## Pipeline Details

### Translation Routing

```
English  → Indic   →  en-indic model (direct)
Indic    → English →  indic-en model (direct)
Indic A  → Indic B →  indic-en → English → en-indic (pivot)
```

### OCR Layers

```
Layer 1: PaddleOCR         — primary, confidence-aware
Layer 2: Tesseract          — fallback if conf < 0.60
Layer 3: Qwen2.5 via Ollama — LLM correction of garbled text
```

### TTS Routing

```
Primary : Indic Parler TTS  — neural, natural voice
Fallback: eSpeak-NG         — all languages, fast, robotic
LLM     : Qwen enhances text before TTS if Whisper conf < 0.60
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | tiny/base/small/medium |
| `INDICTRANS2_EN_INDIC` | `ai4bharat/indictrans2-en-indic-dist-200M` | HF model |
| `INDICTRANS2_INDIC_EN` | `ai4bharat/indictrans2-indic-en-dist-200M` | HF model |
| `M2M100_MODEL` | `facebook/m2m100_418M` | Fallback (commented) |
| `INDIC_PARLER_MODEL` | `ai4bharat/indic-parler-tts` | TTS model |
| `OCR_CONFIDENCE_THRESHOLD` | `0.60` | Below this → Tesseract fallback |
| `OCR_LLM_CORRECTION` | `true` | Enable Qwen OCR correction |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Qwen endpoint |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Qwen model |
| `TTS_LOW_QUALITY_THRESHOLD` | `0.60` | Whisper conf below this → LLM TTS enhancement |
| `UPLOAD_DIR` | `/tmp/ai_translator_uploads` | Temp file storage |
| `OUTPUT_DIR` | `/tmp/ai_translator_output` | TTS output storage |

---

## Enabling M2M-100 Fallback

In `services/translation/translation_service.py`, uncomment:
1. The `M2M100Translator` class definition
2. The `_get_m2m100()` method  
3. The fallback block inside `translate()`

Then set `TRANSLATION_PRIMARY=m2m100` in your environment.

---

## Project Structure

```
ai_translator_api/
├── main.py                          ← FastAPI app + lifespan
├── requirements.txt
├── README.md
├── core/
│   ├── config.py                    ← All settings (env-overridable)
│   └── registry.py                  ← Lazy service loader (singleton)
├── api/
│   └── routes/
│       ├── health.py                ← GET /health*
│       ├── text.py                  ← POST /text/translate
│       ├── ocr.py                   ← POST /ocr/extract*
│       └── speech.py                ← POST /speech/transcribe*
├── services/
│   ├── translation/
│   │   └── translation_service.py   ← IndicTrans2 + M2M-100 fallback
│   ├── ocr/
│   │   └── ocr_service.py           ← Wraps existing OCREngine
│   ├── stt/
│   │   └── stt_service.py           ← Wraps WhisperSTT
│   └── tts/
│       └── tts_service.py           ← Indic Parler + eSpeak-NG
├── models/
│   └── schemas.py                   ← All Pydantic request/response models
└── utils/
    └── language_detector.py         ← Unicode + langdetect detection
```
