# Production deployment

This application is offline at runtime. A deployment is complete only after all
Python packages and model artifacts are installed locally.

## Supported host

- 64-bit Debian/Ubuntu Linux
- Python 3.10 or newer
- `x86_64`, or Raspberry Pi OS 64-bit on `aarch64`
- Raspberry Pi 5 with 8 GB RAM recommended
- Chromium for kiosk mode

PaddlePaddle is the limiting Raspberry Pi dependency. The pinned 2.6.2 package
is installed automatically on x86_64, while ARM64 wheel availability depends on
the Python version and distribution source. On a Raspberry Pi, supply an
explicitly validated ARM64 wheel through
`PADDLE_WHEEL=/absolute/path/to/paddlepaddle.whl`.

## 1. Install the runtime

```bash
git clone <repository-url> ai-translator
cd ai-translator
chmod +x deploy/*.sh scripts/verify_deployment.py
PADDLE_WHEEL=/absolute/path/to/paddlepaddle-arm64.whl ./deploy/install.sh
```

Omit `PADDLE_WHEEL` on x86_64. The installer creates `venv`, installs system
packages, creates the writable upload directory, and creates `.env`.

## 2. Provision offline models

No service downloads a model at runtime. Copy already approved artifacts into
the paths configured in `.env`.

### CTranslate2 IndicTrans2

Both directories are mandatory:

```text
models/indictrans2-en-indic-dist-200M-ct2-int8/
models/indictrans2-indic-en-dist-200M-ct2-int8/
```

Each directory must contain:

```text
model.bin
model.SRC        (or vocab/model.SRC)
model.TGT        (or vocab/model.TGT)
```

Use the production converted artifacts. The deployment process does not
download or reconvert them.

### Faster Whisper

Set `WHISPER_MODEL_PATH` to a complete local Faster Whisper snapshot containing
at least:

```text
model.bin
config.json
tokenizer.json
```

Downloading is a provisioning-time operation only. For example, on a connected
staging machine:

```bash
huggingface-cli download Systran/faster-whisper-base \
  --local-dir models/faster-whisper-base
```

Copy that directory to the offline target.

### PaddleOCR

Set `OCR_MODEL_ROOT` to a local PaddleOCR `whl` tree:

```text
whl/
  rec/<family>/<model>/inference.pdmodel
  rec/<family>/<model>/inference.pdiparams
  rec/<family>/<model>/inference.pdiparams.info
  det/en/<model>/...
  det/ml/<model>/...
  cls/<model>/...
```

Recognition families currently required for the complete catalog are `en`,
`ta`, `devanagari`, `te`, and `ka`. PaddleOCR runtime downloads are disabled;
copy the verified model tree from staging.

### Piper

Set `PIPER_VOICES_DIR`, or provide an explicit JSON `PIPER_VOICE_MAP`. Every
voice requires adjacent files:

```text
voice.onnx
voice.onnx.json
```

If multiple files match one language prefix, use `PIPER_VOICE_MAP` to select
one. For full readiness, provision one voice for every TTS-supported language.
Piper's provisioning utility can download available public voices on a
connected staging machine:

```bash
python -m piper.download_voices --data-dir models/piper <voice-name>
```

Copy the completed voice directory to the offline target.

## 3. Validate before startup

Load configuration and run the deployment gate:

```bash
set -a
source .env
set +a
venv/bin/python scripts/verify_deployment.py
```

The command exits nonzero if Python, architecture, writable storage, a runtime
package, or any required model family is missing. Use `--json` for automation.

## 4. Run manually

Use separate terminals:

```bash
./deploy/start-backend.sh
./deploy/start-frontend.sh
```

Then open `http://127.0.0.1:8080/`. Check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/status
```

`/health` is a liveness endpoint and remains HTTP 200 while reporting the
artifact availability map. `/health/status` reports `degraded` until all
required artifacts are present.

## 5. Install systemd services

Replace the template markers and install:

```bash
PROJECT_DIR="$(pwd)"
DEPLOY_USER="$(id -un)"
sed -e "s|@PROJECT_DIR@|${PROJECT_DIR}|g" -e "s|@USER@|${DEPLOY_USER}|g" \
  deploy/systemd/ai-translator-backend.service \
  | sudo tee /etc/systemd/system/ai-translator-backend.service >/dev/null
sed -e "s|@PROJECT_DIR@|${PROJECT_DIR}|g" -e "s|@USER@|${DEPLOY_USER}|g" \
  deploy/systemd/ai-translator-frontend.service \
  | sudo tee /etc/systemd/system/ai-translator-frontend.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now ai-translator-backend ai-translator-frontend
```

Logs are written to journald and inherit the host's journal rotation policy:

```bash
journalctl -u ai-translator-backend -f
```

Both services restart after crashes. SIGTERM triggers the FastAPI shutdown
lifecycle and unloads model resources.

## 6. Kiosk

Start a graphical Raspberry Pi session, confirm camera/microphone/speaker
permissions, then run:

```bash
./deploy/start-kiosk.sh
```

For automatic graphical login, invoke this script from the desktop
environment's autostart facility. Keep the kiosk URL on `127.0.0.1`; Chromium
treats localhost as a secure context for camera and microphone access.

## Operational checks

- Ensure the service user can write `UPLOAD_DIR`.
- Reserve enough disk for two translation models, Whisper, all OCR families,
  Piper voices, the virtual environment, and OS logs.
- Use a 64-bit OS and active cooling on Raspberry Pi 5.
- Verify camera, microphone, speakers, and touchscreen on the target hardware.
- Configure journald retention according to available disk.
- Graceful shutdown should stop Chromium, frontend, and backend before power
  removal.
