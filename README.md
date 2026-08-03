# AI Translator

AI Translator is a local-first multilingual application for text, image, and
speech translation. It includes a FastAPI backend and a static browser
frontend.

The runtime pipeline uses:

- **IndicTrans2**, converted to CTranslate2, for translation
- **PaddleOCR** for image text extraction
- **Faster Whisper** for speech recognition
- **Piper** for optional speech synthesis

All inference services load local artifacts lazily on first use. They do not
download models at runtime.

## Implemented workflows

- Translate text with automatic or explicit source-language selection.
- Detect the language of text.
- Extract text from an uploaded image.
- Extract and translate text from an uploaded image, with optional Piper audio.
- Transcribe an uploaded or browser-recorded audio file.
- Transcribe and translate speech, with optional Piper audio.
- Report process liveness, loaded services, and local artifact readiness.
- Serve a browser interface with image upload/camera capture, audio
  upload/recording, text input, request cancellation, and client-side file
  validation.

## Architecture

```text
Browser frontend (http://127.0.0.1:8080)
    |
    | HTTP / multipart uploads
    v
FastAPI application (http://127.0.0.1:8000)
    |
    +-- /api/v1/text
    |     +-- language detection
    |     +-- IndicTrans2 translation
    |     `-- optional Piper synthesis
    |
    +-- /api/v1/ocr
    |     +-- bounded temporary image upload
    |     +-- PaddleOCR extraction
    |     +-- optional IndicTrans2 translation
    |     `-- optional Piper synthesis
    |
    +-- /api/v1/speech
    |     +-- bounded temporary audio upload
    |     +-- Faster Whisper transcription
    |     +-- optional IndicTrans2 translation
    |     `-- optional Piper synthesis
    |
    `-- ServiceRegistry
          +-- one shared lazy service instance per subsystem
          +-- thread-safe model initialization
          `-- graceful model release during shutdown
```

Translation routing is:

```text
English -> Indic language: en-indic model
Indic language -> English: indic-en model
Indic language -> Indic language: indic-en, then en-indic
Same source and target: passthrough
```

## Requirements

- 64-bit Linux
- Python 3.10 or newer
- `x86_64`, `aarch64`, or `arm64`
- Local model artifacts for every service you intend to use
- Chromium only if kiosk mode is required

The provided installer targets Debian/Ubuntu. On `x86_64`, it installs the
pinned PaddlePaddle package from `requirements.txt`. On ARM64, provide a
validated PaddlePaddle wheel with `PADDLE_WHEEL`.

## Installation

### Debian/Ubuntu installer

```bash
git clone <repository-url> ai-translator
cd ai-translator
chmod +x deploy/*.sh scripts/verify_deployment.py
./deploy/install.sh
```

For ARM64:

```bash
PADDLE_WHEEL=/absolute/path/to/paddlepaddle.whl ./deploy/install.sh
```

The installer creates `venv`, installs the system and Python dependencies,
creates `/var/tmp/ai-translator/uploads`, and generates `.env` from
`.env.example`.

### Manual Python installation

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sed "s|@PROJECT_DIR@|$(pwd)|g" .env.example > .env
```

Installing Python packages is not enough to run inference. Provision the local
artifacts described below and update `.env` before starting the backend.

## Local model artifacts

### IndicTrans2 translation

Set both `INDICTRANS2_EN_INDIC` and `INDICTRANS2_INDIC_EN`. Each configured
directory must contain:

```text
model.bin
model.SRC        # or vocab/model.SRC
model.TGT        # or vocab/model.TGT
```

The backend uses the CTranslate2 runtime, SentencePiece models, and
`IndicTransToolkit.IndicProcessor`.

### Faster Whisper speech recognition

Set `WHISPER_MODEL_PATH` to a complete local Faster Whisper model directory
containing at least:

```text
model.bin
config.json
tokenizer.json
```

If `WHISPER_MODEL_PATH` is empty, the service searches the local Hugging Face
cache for the configured `WHISPER_MODEL_SIZE`. It still does not download a
model.

### PaddleOCR

Set `OCR_MODEL_ROOT` to a local PaddleOCR model tree:

```text
whl/
  rec/<family>/<model>/inference.pdmodel
  rec/<family>/<model>/inference.pdiparams
  rec/<family>/<model>/inference.pdiparams.info
  det/en/<model>/...
  det/ml/<model>/...
  cls/<model>/...
```

The configured recognition families are `en`, `ta`, `devanagari`, `te`, and
`ka`. Runtime model downloads are disabled.

### Piper speech synthesis

Set `PIPER_VOICES_DIR` to a directory of local voices or set
`PIPER_VOICE_MAP` to a JSON object that maps language codes to voice files.
Every voice requires an adjacent configuration file:

```text
voice.onnx
voice.onnx.json
```

Automatic directory matching expects language-prefixed filenames such as
`en_*.onnx`, `ta_*.onnx`, or `hi_*.onnx`. If more than one voice matches a
language, use `PIPER_VOICE_MAP` to select one explicitly.

## Validate the deployment

Load `.env` and run the readiness gate:

```bash
set -a
source .env
set +a
venv/bin/python scripts/verify_deployment.py
```

Use `--json` for machine-readable output. The verifier returns a nonzero exit
code if the Python version, architecture, upload directory, runtime packages,
or required model artifacts are not ready.

## Startup

Start the backend and frontend in separate terminals:

```bash
./deploy/start-backend.sh
```

```bash
./deploy/start-frontend.sh
```

Open <http://127.0.0.1:8080/>. API documentation is available at
<http://127.0.0.1:8000/docs>.

To run only the backend manually:

```bash
set -a
source .env
set +a
cd backend
../venv/bin/python -m ai_translator_api.run
```

`HOST`, `PORT`, and `RELOAD` control the Uvicorn process.

## Deployment

Production templates are provided for systemd:

```text
deploy/systemd/ai-translator-backend.service
deploy/systemd/ai-translator-frontend.service
```

The backend and frontend services restart after failure and write logs to
journald. The backend handles termination through the FastAPI lifespan and
unloads initialized services during shutdown.

For Chromium kiosk mode after both services are running:

```bash
./deploy/start-kiosk.sh
```

Detailed host preparation, model provisioning, systemd installation, and kiosk
instructions are in [DEPLOYMENT.md](DEPLOYMENT.md).

## API reference

### Text translation

`POST /api/v1/text/translate`

```json
{
  "text": "Hello, how are you?",
  "target_lang": "tam_Taml",
  "source_lang": null,
  "tts": {
    "enabled": true,
    "engine": "auto",
    "speed": 150,
    "return_audio": true
  }
}
```

`source_lang` may be omitted or `null` for automatic detection. The implemented
TTS engine values are `auto` and `piper`. Set `tts.enabled` to `false` when no
Piper output is required.

Example response:

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
    "engine_used": "piper",
    "audio_base64": "UklGRi...",
    "audio_format": "wav",
    "sample_rate": 22050,
    "processing_ms": 325
  },
  "processing_ms": 812
}
```

The actual Piper sample rate comes from the selected voice.

### Text language detection

`POST /api/v1/text/detect-language`

```json
{
  "text": "நான் தமிழ் பேசுகிறேன்"
}
```

Example response:

```json
{
  "detected_lang": "tam_Taml",
  "detected_lang_name": "Tamil",
  "confidence": 0.99
}
```

### OCR extraction

`POST /api/v1/ocr/extract`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ocr/extract \
  -F file=@document.png \
  -F source_lang_hint=tam_Taml
```

`source_lang_hint` is optional. Use one of the language codes marked as OCR
supported below; other configured language codes pass request validation but
cannot select a PaddleOCR recognition model.

Example response:

```json
{
  "success": true,
  "extracted_text": "வணக்கம்",
  "detected_lang": "tam_Taml",
  "detected_lang_name": "Tamil",
  "ocr_confidence": 0.94,
  "ocr_engine_used": "paddle",
  "llm_corrected": false,
  "processing_ms": 240
}
```

`llm_corrected` remains in the response contract and is always `false` in the
current PaddleOCR-only implementation.

### OCR extraction and translation

`POST /api/v1/ocr/extract-and-translate`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ocr/extract-and-translate \
  -F file=@document.png \
  -F target_lang=eng_Latn \
  -F source_lang_hint=tam_Taml \
  -F tts_enabled=false
```

Optional form fields are `source_lang_hint`, `tts_enabled`, `tts_engine`,
`tts_return_audio`, and `tts_speed`. `tts_engine` accepts `auto` or `piper`.

Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`,
`.tiff`, and `.webp`. The default upload limit is 10 MiB.

### Speech transcription

`POST /api/v1/speech/transcribe`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/speech/transcribe \
  -F file=@speech.wav
```

Example response:

```json
{
  "success": true,
  "text": "hello",
  "detected_lang": "eng_Latn",
  "detected_lang_name": "English",
  "whisper_lang": "en",
  "confidence": 0.98,
  "processing_ms": 510
}
```

### Speech transcription and translation

`POST /api/v1/speech/transcribe-and-translate`

```bash
curl -X POST http://127.0.0.1:8000/api/v1/speech/transcribe-and-translate \
  -F file=@speech.wav \
  -F target_lang=tam_Taml \
  -F tts_enabled=false
```

Optional form fields are `tts_enabled`, `tts_engine`, `tts_return_audio`, and
`tts_speed`. Supported audio extensions are `.wav`, `.mp3`, `.ogg`, `.flac`,
`.m4a`, and `.aac`. The default upload limit is 25 MiB and the default maximum
duration is 300 seconds.

### Health and readiness

| Endpoint | Behavior |
|---|---|
| `GET /health` | Liveness. Returns HTTP 200 while the process is alive and includes loaded-service and model-availability maps. |
| `GET /health/status` | Detailed loaded and available state. Returns `status: "degraded"` while required artifacts are missing. |
| `GET /health/languages` | Returns the configured language-code and display-name catalog. |

Expected error responses use one JSON envelope:

```json
{
  "success": false,
  "error": {
    "type": "TranslationModelUnavailableError",
    "category": "model_unavailable",
    "message": "Required local model artifacts are unavailable"
  }
}
```

## Supported languages

Translation, Faster Whisper mapping, and Piper voice routing are configured for
all languages below. Piper synthesis succeeds only when a matching local voice
has been provisioned. PaddleOCR is configured only for the rows marked `Yes` in
the OCR column.

| Code | Language | Translation | Speech recognition | Piper | OCR |
|---|---|---:|---:|---:|---:|
| `eng_Latn` | English | Yes | Yes | Yes | Yes |
| `tam_Taml` | Tamil | Yes | Yes | Yes | Yes |
| `hin_Deva` | Hindi | Yes | Yes | Yes | Yes |
| `tel_Telu` | Telugu | Yes | Yes | Yes | Yes |
| `kan_Knda` | Kannada | Yes | Yes | Yes | Yes |
| `mal_Mlym` | Malayalam | Yes | Yes | Yes | No |
| `ben_Beng` | Bengali | Yes | Yes | Yes | No |
| `guj_Gujr` | Gujarati | Yes | Yes | Yes | No |
| `mar_Deva` | Marathi | Yes | Yes | Yes | Yes |
| `pan_Guru` | Punjabi | Yes | Yes | Yes | No |
| `ory_Orya` | Odia | Yes | Yes | Yes | No |
| `urd_Arab` | Urdu | Yes | Yes | Yes | No |
| `asm_Beng` | Assamese | Yes | Yes | Yes | No |

## Environment variables

`.env.example` contains the deployment-oriented settings. The application also
accepts the tuning variables below.

### Process and storage

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Backend bind address. |
| `PORT` | `8000` | Backend port. |
| `RELOAD` | `false` | Enable Uvicorn reload mode. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed browser origins. |
| `UPLOAD_DIR` | `/tmp/ai_translator_uploads` | Temporary upload directory. |
| `UPLOAD_CHUNK_SIZE_BYTES` | `1048576` | Streamed upload chunk size. |
| `OCR_MAX_UPLOAD_SIZE_BYTES` | `10485760` | Maximum image upload size. |
| `SPEECH_MAX_UPLOAD_SIZE_BYTES` | `26214400` | Maximum audio upload size. |

### IndicTrans2

| Variable | Default | Purpose |
|---|---|---|
| `INDICTRANS2_EN_INDIC` | `models/indictrans2-en-indic-dist-200M-ct2-int8` | Local English-to-Indic CTranslate2 directory. |
| `INDICTRANS2_INDIC_EN` | `models/indictrans2-indic-en-dist-200M-ct2-int8` | Local Indic-to-English CTranslate2 directory. |
| `TRANSLATION_COMPUTE_TYPE` | `int8` | CTranslate2 compute type. |
| `TRANSLATION_MAX_INPUT_TOKENS` | `256` | Maximum encoded input length. |
| `TRANSLATION_MAX_OUTPUT_TOKENS` | `256` | Maximum decoding length. |
| `TRANSLATION_NUM_BEAMS` | `5` | Translation beam size. |
| `TRANSLATION_NO_REPEAT_NGRAM_SIZE` | `2` | Repeated n-gram restriction. |
| `TRANSLATION_BATCH_SIZE` | `8` | Service batch chunk size. |
| `TRANSLATION_INTER_THREADS` | `2` | CTranslate2 inter-op threads. |
| `TRANSLATION_INTRA_THREADS` | `2` | CTranslate2 intra-op threads. |

### Faster Whisper

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_PATH` | empty | Preferred local model directory. |
| `WHISPER_MODEL_SIZE` | `base` | Local cache model name when no path is supplied. |
| `WHISPER_DEVICE` | `cpu` | Inference device. |
| `WHISPER_COMPUTE_TYPE` | `int8` | Faster Whisper compute type. |
| `WHISPER_CPU_THREADS` | `4` | CPU inference threads. |
| `WHISPER_NUM_WORKERS` | `1` | Faster Whisper workers. |
| `WHISPER_BEAM_SIZE` | `5` | Transcription beam size. |
| `WHISPER_BEST_OF` | `5` | Candidate count. |
| `WHISPER_VAD_FILTER` | `true` | Enable voice-activity filtering. |
| `WHISPER_MIN_SILENCE_MS` | `500` | VAD minimum silence duration. |
| `STT_MAX_DURATION_SECONDS` | `300` | Maximum accepted audio duration. |
| `STT_MIN_SAMPLE_RATE` | `8000` | Minimum accepted sample rate. |
| `STT_MAX_SAMPLE_RATE` | `192000` | Maximum accepted sample rate. |
| `STT_BATCH_SIZE` | `4` | Service batch chunk size. |
| `STT_SUPPORTED_EXTENSIONS` | `.wav,.mp3,.ogg,.flac,.m4a,.aac` | Accepted audio extensions. |

### PaddleOCR

| Variable | Default | Purpose |
|---|---|---|
| `OCR_MODEL_ROOT` | `~/.paddleocr/whl` | Local model tree. |
| `OCR_CONFIDENCE_THRESHOLD` | `0.60` | Adds a warning when average confidence is lower. |
| `OCR_USE_ANGLE_CLASSIFIER` | `true` | Enable the configured angle classifier. |
| `OCR_ENABLE_MKLDNN` | `true` | Enable PaddleOCR MKL-DNN execution. |
| `OCR_CPU_THREADS` | `4` | PaddleOCR CPU threads. |
| `OCR_BATCH_SIZE` | `4` | Service batch chunk size. |
| `OCR_SUPPORTED_EXTENSIONS` | `.jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp` | Accepted image extensions. |

### Piper

| Variable | Default | Purpose |
|---|---|---|
| `PIPER_VOICES_DIR` | empty | Directory searched for local voices. |
| `PIPER_VOICE_MAP` | empty | JSON mapping of language codes to `.onnx` files. |
| `PIPER_LENGTH_SCALE` | `1.0` | Piper synthesis length scale. |
| `PIPER_NOISE_SCALE` | `0.667` | Piper synthesis noise scale. |
| `PIPER_NOISE_W_SCALE` | `0.8` | Piper phoneme-width noise scale. |
| `PIPER_VOLUME` | `1.0` | Output volume multiplier. |
| `TTS_MAX_TEXT_CHARS` | `2000` | Maximum synthesis input length. |
| `TTS_BATCH_SIZE` | `4` | Service batch chunk size. |
| `TTS_MIN_SPEED` | `80` | Minimum accepted API speed value. |
| `TTS_DEFAULT_SPEED` | `150` | Default API speed value. |
| `TTS_MAX_SPEED` | `350` | Maximum accepted API speed value. |
| `TTS_LOW_QUALITY_THRESHOLD` | `0.60` | Speech confidence threshold passed as a synthesis hint; the current Piper service does not alter output from this hint. |

## Project structure

```text
.
├── backend/
│   `-- ai_translator_api/
│       ├── api/
│       │   ├── error_handlers.py
│       │   `-- routes/
│       ├── core/
│       │   ├── config.py
│       │   ├── errors.py
│       │   ├── lifecycle.py
│       │   ├── readiness.py
│       │   `-- registry.py
│       ├── models/schemas.py
│       ├── services/
│       │   ├── ocr/ocr_service.py
│       │   ├── stt/stt_service.py
│       │   ├── translation/translation_service.py
│       │   `-- tts/tts_service.py
│       ├── utils/
│       │   ├── language_detector.py
│       │   ├── languages.py
│       │   `-- uploads.py
│       ├── main.py
│       `-- run.py
├── deploy/
│   ├── systemd/
│   ├── install.sh
│   ├── start-backend.sh
│   ├── start-frontend.sh
│   `-- start-kiosk.sh
├── frontend/
│   ├── css/
│   ├── js/
│   `-- index.html
├── scripts/verify_deployment.py
├── .env.example
├── DEPLOYMENT.md
├── README.md
`-- requirements.txt
```
