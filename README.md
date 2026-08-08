# Disaster Response Vision

> **End-to-end computer-vision pipeline** for detecting people and animals in disaster-zone imagery — built as a production-grade rebuild of a university prototype.

[![CI](https://github.com/Lohithaswin/Life-detection-in-Disaster-zones/actions/workflows/ci.yml/badge.svg)](https://github.com/Lohithaswin/Life-detection-in-Disaster-zones/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What it does

Disaster Response Vision ingests images or video streams, runs **COCO-pretrained YOLOv5 / YOLOv8** inference via the Ultralytics API, filters for life-sign classes (person, dog, cat, bird, and 7 other animals), persists every detection to a relational database, and fires **deduplicated async email alerts** when life signs are found. The entire pipeline is accessible three ways: a **Typer CLI**, a **FastAPI REST service**, and a legacy interactive menu preserved from the original project.

This is an honest portfolio rebuild — benchmark metrics are computed by `evaluation/benchmark.py` running `model.val()` on a CC-BY-4.0 licensed dataset. No numbers are hand-typed. Models are **not** fine-tuned for disaster zones; see [Limitations](#limitations).

---

## Key engineering features

| Feature | Implementation |
|---|---|
| **Unified model interface** | Single `Detector` class switches between YOLOv5 and YOLOv8 weights by filename; lazy-loads model on first call |
| **Structured detections** | Frozen `Detection` + `BoundingBox` dataclasses — typed, hashable, serialisable |
| **Async email alerts** | `aiosmtplib`-based async send with SHA-256-keyed in-memory deduplication; configurable suppression window |
| **Dual database support** | SQLAlchemy 2.0 ORM — defaults to SQLite (zero setup), swaps to PostgreSQL via `DATABASE_URL` env var |
| **Schema migrations** | Alembic migration `001_initial_schema` covers `media_records` + `detection_records` tables |
| **REST API** | FastAPI with Pydantic v2 request/response schemas, background alert tasks, and auto `/docs` |
| **CLI** | Typer with `detect`, `alert-test`, and `legacy-menu` subcommands; discovers media recursively |
| **One-command Docker** | `docker compose up --build` starts API + PostgreSQL with health-checks and volume persistence |
| **Real benchmark** | `model.val()` on Ultralytics COCO128 (CC BY 4.0); latency measured per-image after warm-up; charts generated from JSON — no hardcoded numbers |
| **Test suite** | pytest unit tests for dedup logic, mocked-YOLO detector, and in-memory-SQLite repository |
| **CI/CD** | GitHub Actions matrix (Python 3.11 + 3.12): ruff lint → mypy type-check → pytest with coverage |

---

## Architecture

```mermaid
flowchart TD
    subgraph Inputs
        A1[Image file / directory]
        A2[Video file / stream]
        A3[REST upload]
    end

    subgraph Core["disaster_vision package"]
        B["Detector\n(YOLOv5 / YOLOv8)"]
        C["VideoProcessor\n(OpenCV frame loop)"]
        D["AlertDeduplicator\n(SHA-256 keyed, time-windowed)"]
        E["EmailAlerter\n(async aiosmtplib)"]
        F["DetectionRepository\n(SQLAlchemy 2.0)"]
    end

    subgraph Interfaces
        G["CLI  disaster-vision detect"]
        H["FastAPI  POST /detect/*\nGET /detections\nGET /health"]
    end

    subgraph Storage
        I[(SQLite\nlocal dev)]
        J[(PostgreSQL\nDocker / prod)]
    end

    A1 & A2 --> G --> B
    A3 --> H --> B
    B --> C
    B --> F
    B --> D --> E
    F --> I
    F --> J
    F --> H
```

---

## Project structure

```
disaster-response-vision/
├── src/disaster_vision/
│   ├── config.py               # pydantic-settings — all config from env
│   ├── logging_config.py       # structured root-logger setup
│   ├── cli.py                  # Typer CLI entry point
│   ├── legacy_menu.py          # preserved original menu (img/vid/alert)
│   ├── detection/
│   │   ├── detector.py         # Detector, Detection, BoundingBox, LIFE_CLASSES
│   │   └── video.py            # VideoProcessor (OpenCV frame loop)
│   ├── alerts/
│   │   ├── dedup.py            # AlertDeduplicator (configurable window)
│   │   └── email_alert.py      # EmailAlerter — sync + async SMTP, AlertPayload
│   ├── db/
│   │   ├── models.py           # SQLAlchemy ORM: MediaRecord, DetectionRecord
│   │   └── repository.py       # DetectionRepository — single data-access layer
│   ├── api/
│   │   ├── main.py             # FastAPI app, lifespan, background tasks
│   │   └── schemas.py          # Pydantic v2 request/response models
│   └── evaluation/
│       └── benchmark.py        # Real YOLO benchmark — no hardcoded metrics
├── tests/
│   ├── test_dedup.py           # Deduplication logic unit tests
│   ├── test_detector.py        # Detector with mocked YOLO
│   └── test_repository.py      # Repository layer (in-memory SQLite)
├── alembic/versions/
│   └── 001_initial_schema.py   # Initial DB migration
├── scripts/
│   ├── download_weights.py     # Fetch YOLOv5s + YOLOv8n via Ultralytics
│   └── download_sample_data.py # Download CC-licensed COCO sample images
├── .github/workflows/ci.yml    # GitHub Actions: lint → type-check → test
├── Dockerfile                  # python:3.11-slim, installs package + weights
├── docker-compose.yml          # API + PostgreSQL, one-command deploy
├── pyproject.toml              # PEP 517 build, entry-points, ruff, mypy config
└── .env.example                # All config keys documented, no real values
```

---

## Quickstart

### Option A — Docker (recommended, zero setup)

```bash
cd disaster-response-vision
cp .env.example .env        # optional: fill in SMTP_ vars for email alerts
docker compose up --build
```

API live at **http://localhost:8000** · Interactive docs at **http://localhost:8000/docs**

### Option B — Local Python

```bash
cd disaster-response-vision

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env
python scripts/download_weights.py          # fetches yolov8n.pt + yolov5s.pt
python scripts/download_sample_data.py      # downloads 3 CC-licensed sample images

# Run detection on sample images (no DB, no alerts)
disaster-vision detect data/samples --no-alert --no-db

# Start the API server
disaster-vision-api
```

### Apply DB migrations

```bash
alembic upgrade head
```

---

## CLI reference

```bash
# Detect in an image, video, or directory (recursive)
disaster-vision detect <source> [--model yolov8n] [--life-only] [--no-alert] [--no-db]

# Send a test alert email (validates SMTP config)
disaster-vision alert-test

# Original interactive menu
disaster-vision legacy-menu
```

**Example — detect only people/animals in all images under a folder:**
```bash
disaster-vision detect data/samples --life-only --model yolov5s
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns `{"status": "ok", "version": "..."}` |
| `GET` | `/detections?limit=100` | Recent detection records from DB |
| `POST` | `/detect/image` | Upload image (multipart/form-data) → detections + optional alert |
| `POST` | `/detect/video` | Upload video (multipart/form-data) → detections + optional alert |

**POST `/detect/image` form fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | — | Image file (jpg/png) |
| `model` | string | `yolov8n` | Model name or weight filename |
| `life_only` | bool | `false` | Filter to person/animal classes only |
| `send_alert` | bool | `true` | Send email if life signs detected |
| `persist` | bool | `true` | Write results to database |

Full interactive docs: `http://localhost:8000/docs`

---

## Configuration

All settings are read from environment variables (or a `.env` file). Copy `.env.example` to `.env` and edit:

```ini
# Detection
DEFAULT_MODEL=yolov8n
CONFIDENCE_THRESHOLD=0.25

# Database — SQLite (local) or PostgreSQL (production)
DATABASE_URL=sqlite:///./data/disaster_vision.db
# DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/disaster_vision

# Email alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_TO=recipient@example.com
ALERT_DEDUP_SECONDS=300   # suppress repeat alerts within 5 minutes
```

---

## Benchmark

Run the benchmark to generate real metrics (requires `.[benchmark]` extras):

```bash
pip install -e ".[benchmark]"
python scripts/download_weights.py --models yolov8n.pt yolov5s.pt
python -m disaster_vision.evaluation.benchmark
```

- **Dataset:** Ultralytics `coco128.yaml` — 128-image subset of MS COCO train2017
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Method:** `model.val()` for mAP metrics; per-image latency measured over 15 runs after 3-run warm-up
- **Output:** `evaluation/results/benchmark_results.json` + bar chart, mAP comparison, correlation heatmap

> Benchmark results are not committed because they depend on your hardware. Run the command above to produce them locally. The script prints a summary table you can paste directly into this README.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=disaster_vision
```

| Test file | What it covers |
|---|---|
| `test_dedup.py` | Deduplication window suppression and `clear()` reset |
| `test_detector.py` | `Detector.detect_image()` with fully mocked YOLO model |
| `test_repository.py` | Insert + list detections via in-memory SQLite |

---

## Tech stack

| Layer | Technology |
|---|---|
| Detection | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLOv5 + YOLOv8), PyTorch 2.0+ |
| Video | OpenCV 4.8+ |
| API | FastAPI 0.100+, Uvicorn, Pydantic v2, python-multipart |
| CLI | Typer |
| Database | SQLAlchemy 2.0 (SQLite / PostgreSQL), Alembic |
| Alerts | aiosmtplib (async SMTP), python-dotenv |
| Config | pydantic-settings |
| Packaging | pyproject.toml (PEP 517/660), setuptools src-layout |
| Testing | pytest, pytest-cov |
| Linting | ruff (E, F, I, UP, B rules) |
| Type-checking | mypy |
| CI | GitHub Actions (matrix: Python 3.11 + 3.12) |
| Deploy | Docker, Docker Compose, PostgreSQL 16 |

---

## Limitations

- **COCO-pretrained only** — models were not fine-tuned on disaster-specific imagery. Accuracy on aerial, rubble-obscured, or thermal images will be lower than the COCO128 benchmark numbers suggest.
- **COCO128 is a small benchmark subset** — 128 images from the training split, not the official COCO val set. Use the numbers for relative model comparison, not as absolute performance claims.
- **No GPU in CI** — latency numbers in the benchmark are CPU-only unless you run it on a CUDA machine.
- **API video upload** — the `/detect/video` endpoint loads the full video into memory; suitable for demos, not production-scale streaming.
- **In-memory deduplication** — alert dedup state resets on process restart. A Redis/DB-backed dedup would be needed for multi-instance deployments.

---

## Security

Old commits (pre-rebuild) contained hardcoded SMTP and database credentials. Those commits have been rewritten with `git filter-repo`. The working tree and current history contain no real secrets. See [`docs/SCRUB_GIT_HISTORY.md`](docs/SCRUB_GIT_HISTORY.md) for methodology.

---

## License

MIT — see [LICENSE](LICENSE).