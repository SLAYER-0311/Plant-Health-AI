---
title: Plant Health AI
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# PlantHealth AI

A deep learning application for plant disease classification using Convolutional Neural Networks (CNNs). This project includes both a custom CNN built from scratch and transfer learning with ResNet50 for improved accuracy.

## Features

- **Custom CNN Model**: 4-block CNN architecture targeting >70% accuracy
- **Transfer Learning**: ResNet50 pretrained model targeting >90% accuracy
- **FastAPI Backend**: RESTful API for model inference
- **React Frontend**: Modern UI for image upload and prediction display
- **Jupyter Notebooks**: Training and evaluation workflows

## Dataset

Uses the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle:
- ~87,000 RGB images
- 38 disease classes across 14 crop species
- Pre-split into training (~70k) and validation (~17k) sets

## Project Structure

```
Plant-Health-AI/
├── config.yaml              # Training configuration
├── requirements.txt         # Python dependencies
├── setup_kaggle.py          # Kaggle authentication setup
├── download_dataset.py      # Dataset download script
├── src/
│   ├── data/                # Dataset and transforms
│   ├── models/              # CNN architectures
│   ├── training/            # Training loop and callbacks
│   ├── evaluation/          # Metrics and evaluation
│   └── utils/               # Visualization utilities
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Custom_CNN.ipynb
│   └── 03_Transfer_Learning.ipynb
├── backend/                 # FastAPI application
│   └── app/
│       ├── main.py
│       ├── routers/
│       ├── schemas/
│       └── services/
└── frontend/                # React application
    └── src/
        ├── components/
        ├── services/
        └── types/
```

## Quick Start

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Kaggle API

```bash
# Run setup script (follow prompts to provide kaggle.json path)
python setup_kaggle.py
```

### 3. Download Dataset

```bash
python download_dataset.py
```

### 4. Train Models

Use the Jupyter notebooks in order:

1. **01_Data_Exploration.ipynb**: Explore the dataset
2. **02_Custom_CNN.ipynb**: Train the custom CNN model
3. **03_Transfer_Learning.ipynb**: Train ResNet50 transfer learning model

Or train from command line (coming soon).

### 5. Run the Application

**Backend (FastAPI):**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (React):**
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000 to use the application.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check and model status |
| `/api/predict` | POST | Upload image for disease prediction |
| `/api/classes` | GET | List all supported disease classes |

## Configuration

Edit `config.yaml` to customize:

```yaml
training:
  batch_size: 32
  num_epochs: 30
  learning_rate: 0.001
  
model:
  custom_cnn:
    dropout_rate: 0.5
  transfer_learning:
    model_name: "resnet50"
    freeze_backbone: true
```

## Disease Classes

The model can classify 38 different plant conditions across 14 crop species:

| Crop | Diseases |
|------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| And more... | See config.yaml for full list |

## Tech Stack

- **ML Framework**: PyTorch
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Vite, React 18, TypeScript, Tailwind CSS
- **Data Processing**: Albumentations, Pillow
- **Visualization**: Matplotlib, Seaborn

## Hardware Requirements

- **Training**: CUDA-compatible GPU recommended (8GB+ VRAM)
- **Inference**: CPU or GPU

## License

MIT License



Here are the commands to run the frontend and backend in separate terminals:

Backend Terminal

cd E:\Projects\Plant-Health-AI
.venv311\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000



Frontend Terminal

cd E:\Projects\Plant-Health-AI\frontend
npm run dev
---

After running both:

- Backend will be at: http://localhost:8000
- Backend API docs: http://localhost:8000/docs
- Frontend will be at: http://localhost:3000


Note: Make sure to run each command in a separate terminal window so both services run simultaneously.

## Hugging Face Space Deployment

This project is deployed on Hugging Face Spaces using Docker. The Space automatically runs both the backend API and serves the frontend UI.

**Live Demo**: https://huggingface.co/spaces/jsohamg/plant-health-ai

### Supported Plants (14 Types):
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### OOD Detection:
The model includes Out-of-Distribution detection to reject non-leaf images like cloth, solid colors, or random objects. It will also warn you if you upload unsupported plant species (e.g., banana, mango).


Frontend commands:

cd frontend
npm install
npm run build
cd ..
.\plant-health-env\Scripts\streamlit.exe run streamlit_app.py

Backend commands:
python app.py
.\plant-health-env\Scripts\python.exe app.py


