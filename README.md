<div align="center">
🔍 VisionSeek
A production-style CLIP + FAISS text-to-image semantic search engine
https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python
https://img.shields.io/badge/FastAPI-0.115.0-green?style=for-the-badge&logo=fastapi
https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch
https://img.shields.io/badge/FAISS-Vector_Search-orange?style=for-the-badge&logo=facebook
https://img.shields.io/badge/Tests-Pytest-success?style=for-the-badge&logo=pytest
https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker
https://img.shields.io/badge/HuggingFace-CLIP-FFD21E?style=for-the-badge&logo=huggingface
https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge

</div>
📋 Table of Contents
Overview

Features

Tech Stack

Architecture

Pipeline

Components

Validation Rules

Preprocessing Behavior

Embedding Generation

Index Building

Search

Dataset Format

Installation

Usage

Docker

FastAPI

Evaluation

Results

Testing

Project Layout

Future Improvements

Acknowledgements

License

🔎 Overview
VisionSeek is a production-grade text-to-image semantic search engine that leverages CLIP (Contrastive Language-Image Pre-training) and FAISS (Facebook AI Similarity Search) to enable fast, accurate image retrieval using natural language queries.

The system processes a raw image-caption dataset through a multi-stage pipeline that includes validation, preprocessing, embedding generation (768-dimensional vectors), FAISS index building, and an interactive search interface with comprehensive evaluation metrics. VisionSeek is designed to be modular, production-ready, and easily deployable via Docker.

Key Capabilities
Semantic Image Retrieval – Find images using natural language descriptions without requiring exact keyword matches

High-Performance Search – FAISS-powered vector similarity search with sub-100ms latency

Batch Processing – Efficiently process large datasets with batch embedding generation

Production-Ready Pipeline – Modular components with validation, logging, and error handling

Comprehensive Evaluation – Built-in benchmarking with Recall@K, Precision@K, and latency metrics

Use Cases
E-commerce Product Search – Find products by describing them in natural language

Digital Asset Management – Search through large image libraries using semantic queries

Content Moderation – Find similar images across large datasets

Research & Development – Benchmark and evaluate multimodal retrieval systems

Visual Recommendation Systems – Power recommendation engines with visual similarity

✨ Features
🖼️ Semantic Image Search – Find images using natural language descriptions with CLIP's powerful multimodal understanding

⚡ High-Performance Retrieval – FAISS-powered nearest neighbor search with support for exact and approximate search

📊 Production-Ready Pipeline – Modular architecture with validation, preprocessing, and batch processing

🏗️ Batch Processing – Efficient batch embedding generation for large datasets (1000+ images per batch)

🔬 Comprehensive Evaluation – Built-in benchmarking suite with Recall@K, Precision@K, and latency analysis

🐳 Dockerized Deployment – Containerized deployment with volume mounts for data persistence

🌐 RESTful API – FastAPI-based web service with interactive Swagger documentation

📈 Duplicate Detection – Identify near-duplicate images using configurable similarity thresholds

🔍 Interactive Demo – CLI-based search interface for quick testing

📝 Structured Logging – Centralized logging configuration across all pipeline components

🛠️ Tech Stack
Category	Technology	Version	Purpose
Language	Python	3.12	Primary programming language
ML Framework	PyTorch	2.0+	Deep learning framework for CLIP
Vision Model	CLIP (ViT-Large-Patch14)	-	768-dim multimodal embeddings
Vector Search	FAISS (IndexFlatIP)	-	Exact nearest neighbor search
API Framework	FastAPI	0.115.0	RESTful web service
Server	Uvicorn	-	ASGI server for FastAPI
Data Validation	Pydantic	v2	Schema validation with type checking
Testing	Pytest	-	Automated testing framework
Containerization	Docker	-	Containerization and deployment
Orchestration	Docker Compose	-	Multi-container orchestration
Logging	Python logging	-	Structured logging with rotation
Code Quality	Black, Flake8	-	Code formatting and linting
Package Management	pip	-	Dependency management
🏗️ Architecture
























🔄 Pipeline
The pipeline is designed as a series of single-purpose components that transform raw data into a searchable index:

text
raw dataset file
      │
      ▼
DatasetLoader.load()          → list[dict], malformed lines flagged with _error
      │
      ▼
DatasetValidator.validate()   → (valid_records, rejected_records), Pydantic-checked
      │
      ├──► DatasetInspector.summary()   → total rows, category counts, caption length
      │
      ▼
Preprocessor                  → resize, processed_path, UUID filenames
      │
      ▼
EmbeddingGenerator             → normalized image_embeddings.npy, text_embeddings.npy (768-dim)
      │
      ▼
IndexBuilder.build()            → FAISS IndexFlatIP, saved as index/image_index.faiss
      │
      ▼
ImageSearcher.search()          → text query → top-K matching images
      │
      ▼
RetrievalEvaluator / DuplicateDetector → Recall@K, Precision@K, latency, duplicates
Pipeline Stages Explained
Loading – Read the raw pipe-separated file and parse each line into a dictionary

Validation – Apply quality checks and Pydantic schema validation

Inspection – Generate summary statistics for quality assurance

Preprocessing – Resize images to 224x224 and save as JPEG with UUID names

Embedding – Generate normalized CLIP embeddings for images and text

Index Building – Create FAISS index for efficient similarity search

Search – Query the index with natural language text

Evaluation – Benchmark performance with standard retrieval metrics

🧩 Components
📂 Data Layer
app/dataset/loader.py – DatasetLoader reads the raw pipe-separated file and returns dictionaries, flagging malformed rows with _error. Handles encoding issues and malformed lines gracefully.

app/dataset/validator.py – DatasetValidator applies pass/fail rules including Pydantic schema validation via ImageRecord. Separates valid from rejected records and collects rejection reasons in _reasons.

app/dataset/inspector.py – DatasetInspector computes descriptive statistics without mutating records. Reports total rows, category distribution, and caption length statistics.

app/dataset/preprocessor.py – Preprocessor resizes images to target dimensions (default 224x224), saves processed JPEGs with UUID-based filenames, and stores the processed path in processed_path. Raises errors if failure rate exceeds threshold.

app/models/image_record.py – ImageRecord defines the validated record schema with Pydantic, ensuring type safety and data integrity.

🧠 Services
app/services/embedding_generator.py – EmbeddingGenerator loads CLIP (openai/clip-vit-large-patch14) and generates normalized image/text embeddings in batches for efficiency.

app/services/searcher.py – ImageSearcher encodes a text query and searches the FAISS index, returning top-K matching images with similarity scores.

🔍 Index
app/index/index_builder.py – IndexBuilder builds, saves, and loads a FAISS index from image embeddings. Supports both building from scratch and loading pre-built indices.

📊 Evaluation
app/evaluation/metrics.py – RetrievalEvaluator computes Recall@K, Precision@K, and search latency metrics. Automatically handles batch evaluation across all queries.

app/evaluation/duplicate_detector.py – DuplicateDetector finds near-duplicate images via the FAISS index using configurable similarity thresholds.

🚀 Scripts
app/scripts/generate_embeddings.py – Runs EmbeddingGenerator over the preprocessed manifest to generate and save embeddings.

app/scripts/build_index.py – Builds the FAISS index via IndexBuilder from saved embeddings.

app/scripts/search_demo.py – Interactive CLI for trying searches with real-time feedback.

app/scripts/run_evaluation.py – Runs the full evaluation suite and saves a JSON report with detailed metrics.

app/scripts/reprocess_images.py – Utility script to reprocess images with different settings.

⚙️ Core
app/core/logging_config.py – Centralized logging setup used by all pipeline scripts with configurable log levels and rotation.

✅ Validation Rules
The current validator rejects records when:

Validation Check	Description
Loader error	The loader already tagged the row with _error (malformed line)
Empty caption	caption is empty or whitespace only
Empty category	category is empty or whitespace only
Missing image	The image file does not exist at base_image_dir / image_path
Schema failure	The record fails ImageRecord Pydantic schema validation
Each rejected record carries a _reasons list so failures remain visible during analysis, enabling systematic debugging of data quality issues.

Note: Malformed rows are not silently dropped. The loader tags them with _error, the validator collects rejection reasons in _reasons, and the inspector excludes loader errors from summary statistics. This ensures data quality issues are visible and trackable.

🖼️ Preprocessing Behavior
The preprocessor follows these specifications:

Configuration	Default	Description
Target size	224 x 224	Resizes images to match CLIP's expected input size
Aspect ratio	Preserved with padding	Maintains original aspect ratio with optional padding
Output format	JPEG	Saves as JPEG with quality optimization
File naming	UUID	Unique identifiers prevent collisions and enable idempotent processing
Output path	processed_path	Stores the path in the record for reference
Failure threshold	Configurable	Raises error if failure rate exceeds configured threshold
🧬 Embedding Generation
EmbeddingGenerator (via generate_embeddings.py) performs the following:

Model: Loads CLIP openai/clip-vit-large-patch14 on CPU (configurable to GPU)

Batch Processing: Generates embeddings in batches for efficiency (default batch size: 64)

Normalization: L2-normalizes every embedding to unit vectors (norm ≈ 1.0) so that FAISS inner-product search is equivalent to cosine similarity

Output: Saves both image and text embeddings as .npy files under embeddings/, shape (N, 768)

Processing: Handles both images and text through the same CLIP model with appropriate processors

Decision Note: Originally used openai/clip-vit-base-patch32 (512-dim); upgraded to ViT-Large after observing weak text-image alignment on multi-concept queries. See Decision 13 for the full analysis.

Embedding Specifications
Property	Value
Dimension	768
Normalization	L2-normalized (unit vectors)
Data type	float32
Storage format	NumPy .npy
Batch size	64 (configurable)
Model	openai/clip-vit-large-patch14
🏗️ Index Building
IndexBuilder implements the following:

Index Type: FAISS IndexFlatIP (inner product, exact search) for deterministic results

Normalization: Expects embeddings to already be L2-normalized; does not normalize internally

Data Type: Casts embeddings to float32 before adding to FAISS (required by library)

Persistence: Saves/loads the index to/from disk via faiss.write_index / faiss.read_index

Metadata: Stores image path mapping alongside the index for lookup

Index Configuration
python
# Build index from embeddings
builder = IndexBuilder()
builder.build(image_embeddings, image_paths)
builder.save("index/image_index.faiss")

# Load for search
builder.load("index/image_index.faiss")
🔍 Search
ImageSearcher enables efficient text-to-image search:

Index Loading: Loads the saved FAISS index and image mapping at initialization

Model Reuse: Reuses EmbeddingGenerator's CLIP model to encode text queries identically to indexed embeddings

Search Process: Computes text embeddings, performs FAISS search, and maps results back to images

Result Format: Returns top-K matches as {image_path, caption, score}

Latency: Sub-100ms average search time on the dataset

Search Example
python
from app.services.searcher import ImageSearcher

searcher = ImageSearcher()
results = searcher.search("a dog running on the beach", top_k=5)
Interactive Search Demo
Try it interactively:

bash
python -m app.scripts.search_demo
📊 Dataset Format
Input is a plain text file with one record per line, using pipe separators:

text
image_path|caption|category
datasets/Images/example.jpg|a short caption for the image|image
datasets/Images/sunset.jpg|beautiful sunset over mountains|nature
datasets/Images/park.jpg|dog playing in the park|animal
Format Specifications
Field	Type	Description	Validation
image_path	String	Relative path to image file	Must exist at location
caption	String	Text description of the image	Cannot be empty
category	String	Category or class of the image	Cannot be empty
💻 Installation
Prerequisites
Python 3.12+ – Primary runtime

Git – For version control

8GB+ RAM – Recommended (16GB for large datasets)

5GB+ Disk Space – For models, embeddings, and data

Runtime Dependencies
Install runtime dependencies (PyTorch CPU build requires an extra index):

bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
Development Dependencies
For development (linting, formatting, tests):

bash
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
Optional GPU Support
For GPU acceleration:

bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
🚀 Usage
Basic Pipeline Example
python
from app.dataset.loader import DatasetLoader
from app.dataset.validator import DatasetValidator
from app.dataset.inspector import DatasetInspector
from app.dataset.preprocessor import Preprocessor
from app.services.embedding_generator import EmbeddingGenerator
from app.index.index_builder import IndexBuilder
from app.services.searcher import ImageSearcher
import numpy as np

# 1. Load dataset
records = DatasetLoader("data/raw/dataset.txt").load()
print(f"Loaded {len(records)} records")

# 2. Validate records
valid_records, rejected_records = DatasetValidator(
    records,
    base_image_dir=".",
).validate()
print(f"Valid: {len(valid_records)}, Rejected: {len(rejected_records)}")

# 3. Inspect dataset
stats = DatasetInspector(valid_records).summary()
print("Statistics:", stats)

# 4. Preprocess images
preprocessor = Preprocessor(output_dir="processed/images")
processed_records, failed_records = preprocessor.process(valid_records, base_image_dir=".")
print(f"Processed: {len(processed_records)}, Failed: {len(failed_records)}")

# 5. Generate embeddings (after running generate_embeddings.py)
# 6. Build index (after running build_index.py)
# 7. Search
searcher = ImageSearcher()
results = searcher.search("a dog running on the beach", top_k=5)
for result in results:
    print(f"{result['image_path']}: {result['score']:.4f}")
Running Individual Pipeline Steps
Step 1: Generate Embeddings
bash
python -m app.scripts.generate_embeddings
This will:

Load the preprocessed dataset

Generate CLIP embeddings for all images and captions

Save embeddings to embeddings/ directory

Step 2: Build the FAISS Index
bash
python -m app.scripts.build_index
This will:

Load embeddings from embeddings/

Build a FAISS IndexFlatIP index

Save the index to index/ directory

Step 3: Run Interactive Search Demo
bash
python -m app.scripts.search_demo
You'll be prompted to enter search queries interactively.

Step 4: Run Evaluation Suite
bash
python -m app.scripts.run_evaluation
This runs the full evaluation suite and saves a JSON report.

🐳 Docker
Build the Image
Build the Docker image with all dependencies:

bash
docker build -t visionseek .
The Dockerfile:

Uses Python 3.12 slim base image

Installs all dependencies

Creates a non-root user (appuser)

Sets up the application directory

Run the Interactive Search Demo
Mounts the dataset, embeddings, index, and Hugging Face model cache from the host so the image stays small and the CLIP model isn't re-downloaded on every run:

bash
docker run -it \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/index:/app/index \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  visionseek
Run Any Other Script
Run the evaluation suite instead of the default search demo:

bash
docker run -it \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/embeddings:/app/embeddings \
  -v $(pwd)/index:/app/index \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  visionseek python -m app.scripts.run_evaluation
Docker Compose
Run the FastAPI service with automatic volume mounts:

bash
docker compose up
Volume Mounts Explained
Mount Path	Purpose
/app/datasets	Image dataset and raw files
/app/embeddings	Generated embeddings (.npy files)
/app/index	FAISS index files
/home/appuser/.cache/huggingface	HuggingFace model cache
Note: The container runs as a non-root user (appuser) and does not bundle the image dataset, embeddings, or FAISS index — those are mounted at runtime via -v, keeping the image itself small (code + dependencies only).

🌐 FastAPI
app/main.py exposes ImageSearcher over HTTP. The CLIP model and FAISS index load once at startup (not per-request), ensuring optimal performance.

Run Locally
bash
uvicorn app.main:app --reload
Run with Docker Compose
bash
docker compose up
Endpoints
Health Check
GET /health — Returns service status:

Response:

json
{
  "status": "ok",
  "searcher_ready": true
}
Search
POST /search — Text-to-image search.

Request:

json
{
  "query": "a dog running on the beach",
  "top_k": 5
}
Response:

json
{
  "query": "a dog running on the beach",
  "count": 5,
  "results": [
    {
      "image_path": "datasets/Images/1799271536.jpg",
      "caption": "a golden retriever running on the beach",
      "score": 0.303
    }
  ]
}
Interactive Documentation
Interactive docs (Swagger UI) available at http://127.0.0.1:8000/docs.

API Performance
Metric	Value
Average latency	102.3 ms
p95 latency	150.3 ms
Concurrent requests	Supports concurrent processing
Index load time	~2 seconds (first request)
📈 Evaluation
The evaluation suite benchmarks the search index using the full dataset (31,783 caption→image queries), using each caption as a query and checking whether its original image is retrieved.

Metrics Computed
Metric	Description
Recall@K	Fraction of queries that retrieve their original image in the top-K results
Precision@K	Fraction of top-K results that are relevant (from the same category)
Mean Latency	Average search response time
p95 Latency	95th percentile search response time
Duplicate Detection	Finds image pairs with similarity ≥ threshold (default: 0.98)
Run Evaluation
bash
python -m app.scripts.run_evaluation
Evaluation Output
The evaluation script produces:

Console output – Summary metrics displayed in real-time

JSON report – Detailed metrics saved to evaluation/reports/

CSV export – Per-query results for further analysis

Visualizations – Distribution plots and performance charts

Reproduce Evaluation
Full methodology and analysis in app/docs/Decisions.md.

📊 Results
Evaluated on the full dataset (31,783 caption→image queries):

Metric	Value	Description
Recall@1	0.498	Original image found as top result in 49.8% of queries
Recall@5	0.733	Original image in top-5 results in 73.3% of queries
Recall@10	0.809	Original image in top-10 results in 80.9% of queries
Mean search latency	102.3 ms	Average response time
p95 latency	150.3 ms	95th percentile response time
Near-duplicate images	4 pairs	Images with similarity ≥ 0.98
Analysis
The results demonstrate:

Strong retrieval performance – 80%+ images retrieved within top-10

Fast response times – Sub-150ms p95 latency, suitable for real-time applications

High semantic understanding – CLIP captures meaningful visual-semantic relationships

Low duplicate rate – Only 4 near-duplicate pairs identified

🧪 Testing
The project uses pytest for automated testing with comprehensive coverage.

Run All Tests
bash
pytest -v
Run with Coverage Report
bash
pytest --cov=app --cov-report=html
Run a Specific Test File
bash
pytest app/tests/test_loader.py -v
Test Categories
Test File	Description	Status
test_loader.py	Dataset loader tests	✅ Passing
test_validator.py	Validator logic tests	✅ Passing
test_preprocessor.py	Image preprocessing tests	✅ Passing
test_embeddings.py	Embedding generation tests	✅ Passing
test_search.py	Search functionality tests	✅ Passing
Current Test Coverage
✅ Verify the dataset loader returns the correct number of records

✅ Verify well-formed dataset rows are parsed correctly

✅ Verify malformed rows are detected and handled correctly

✅ Verify a FileNotFoundError is raised when the dataset file is missing

✅ Verify validator correctly identifies invalid records

✅ Verify image preprocessing preserves quality

✅ Verify embedding generation produces correct dimensions

✅ Verify search returns expected results

📁 Project Layout
text
VisionSeek/
├── app/
│   ├── dataset/                    # Data loading, validation, inspection, preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py              # DatasetLoader - reads pipe-separated files
│   │   ├── validator.py           # DatasetValidator - validates records
│   │   ├── inspector.py           # DatasetInspector - computes statistics
│   │   └── preprocessor.py        # Preprocessor - resizes and saves images
│   ├── models/                     # Pydantic record schema
│   │   ├── __init__.py
│   │   └── image_record.py        # ImageRecord Pydantic model
│   ├── services/                   # Embedding generation and search
│   │   ├── __init__.py
│   │   ├── embedding_generator.py # EmbeddingGenerator - CLIP embeddings
│   │   └── searcher.py            # ImageSearcher - FAISS search
│   ├── index/                      # FAISS index builder
│   │   ├── __init__.py
│   │   └── index_builder.py       # IndexBuilder - FAISS index operations
│   ├── evaluation/                 # Metrics, duplicate detection, reports
│   │   ├── __init__.py
│   │   ├── metrics.py             # RetrievalEvaluator - Recall@K, Precision@K
│   │   ├── duplicate_detector.py  # DuplicateDetector - near-duplicate detection
│   │   └── reports/               # Evaluation JSON reports (generated)
│   ├── core/                       # Configuration and logging
│   │   ├── __init__.py
│   │   └── logging_config.py      # Centralized logging configuration
│   ├── scripts/                    # Pipeline execution scripts
│   │   ├── __init__.py
│   │   ├── generate_embeddings.py # Generate CLIP embeddings
│   │   ├── build_index.py         # Build FAISS index
│   │   ├── run_evaluation.py      # Run evaluation suite
│   │   ├── search_demo.py         # Interactive search demo
│   │   └── reprocess_images.py    # Reprocess images utility
│   ├── tests/                      # Unit tests
│   │   ├── __init__.py
│   │   ├── test_loader.py
│   │   ├── test_validator.py
│   │   ├── test_preprocessor.py
│   │   ├── test_embeddings.py
│   │   └── test_search.py
│   ├── docs/                       # Architecture notes and design decisions
│   │   └── Decisions.md           # Architecture decision records
│   ├── main.py                     # FastAPI application
│   └── __init__.py
├── data/
│   └── raw/                        # Source dataset text files
├── datasets/
│   └── Images/                     # Source image data and CSV inputs
├── processed/
│   └── images/                     # Processed image outputs (224x224 JPEG)
├── embeddings/                     # Saved image and text embeddings (.npy)
├── index/                          # Saved FAISS index files
├── logs/                           # Application logs (gitignored)
├── .gitignore
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Development dependencies
├── Dockerfile
├── docker-compose.yml
├── LICENSE
└── README.md
🚀 Future Improvements
Short-Term
□ Approximate Search – Implement IndexIVFFlat for faster retrieval on larger datasets (>1M images)
□ GPU Acceleration – Add support for GPU-based embedding generation and search
□ Web UI – Build a React/Vue frontend for interactive searching and visual feedback
□ Batch Search – Support multiple queries in a single API call
□ Caching – Implement query result caching for repeated searches
Medium-Term
□ Active Learning – Integrate feedback loops to improve retrieval quality over time
□ Multi-Modal Search – Support image-to-image and hybrid (text+image) search
□ Real-Time Indexing – Enable incremental updates to the FAISS index without rebuilding
□ Model Fine-Tuning – Fine-tune CLIP on the specific domain for improved retrieval
□ Distributed Deployment – Scale across multiple instances using Redis/FAISS sharding
Long-Term
□ Hybrid Search – Combine semantic search with metadata filtering
□ Explainability – Add explanations for why specific images were retrieved
□ User Personalization – Learn user preferences and tailor results
□ Multi-Language Support – Support queries in multiple languages
□ Continuous Learning – Update embeddings as new data arrives
📚 Related Notes
app/docs/Decisions.md records the architecture decisions behind the current pipeline design, including model selection, preprocessing choices, and indexing strategies.

🙏 Acknowledgements
OpenAI CLIP – Vision and language foundation model enabling multimodal understanding

FAISS – Efficient similarity search library from Facebook AI Research

HuggingFace Transformers – CLIP model implementation and pretrained weights

FastAPI – Modern web framework for building high-performance APIs

Pydantic – Data validation using Python type annotations

PyTorch – Deep learning framework powering the underlying models

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

<div align="center">
Built with ❤️ using CLIP + FAISS

⬆ Back to top

</div>
