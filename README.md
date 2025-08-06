
# Handwritten Prescription Recognition using CNN-RNN-CTC + FastAPI Deployment

## Project Goal
The goal of this project is to build an end-to-end deep learning pipeline that accurately recognizes handwritten medicine names from prescription images. The system uses Convolutional Neural Networks (CNN) for feature extraction, Recurrent Neural Networks (RNN) for sequence modeling, and Connectionist Temporal Classification (CTC) for decoding unsegmented text.

This is especially useful for:

- Digitizing medical records
- Assisting pharmacists with poor handwriting
- Minimizing human error in drug dispensing

## Key Components
- Image Preprocessing (resizing, cleaning)
- Label Preprocessing (character map creation, encoding)
- Model (CNN + BiLSTM + CTC Loss)
- Training (with validation + test split)
- Inference Pipeline (image → predicted text)
- FastAPI deployment for real-world interaction
- Modular file structure for scalability & clarity

## ️ File Structure Overview
```bash
prescription_ocr/
├── data/
│   ├── raw/                  # Unprocessed original data
│   │   ├── Training/
│   │   ├── Validation/
│   │   └── Testing/
│   └── processed/            # Cleaned images and encoded labels
│       ├── Training/
│       ├── Validation/
│       └── Testing/
│       └── char_map.json
│       └── train.csv / val.csv / test.csv
│
├── notebooks/
│   └── eda.ipynb             # Exploratory Data Analysis
│
├── scripts/
│   ├── clean_images.py       # Runs image cleaning pipeline
│   ├── clean_val.py          # Cleans validation data
│   ├── clean_test.py         # Cleans test data
│   └── train_model.py        # Entry point for training the model
│
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py     # Image cleaning & resizing
│   │   ├── text_preprocessing.py      # Label encoding, char mapping
│   │   └── __init__.py
│   ├── models/
│   │   └── crnn_ctc_model.py          # CNN + RNN + CTC model definition
│   ├── training/
│   │   └── train_utils.py             # Loss, optimizer, training loop
│   ├── inference/
│   │   └── predictor.py               # Load model & predict text from image
│   └── utils/
│       └── helpers.py                 # Helper functions
│
├── tests/
│   └── test_pipeline.py       # Unit tests for preprocessing/model/inference
│
├── app/
│   └── main.py                # FastAPI app for serving model predictions
│
├── logs/
│   └── train.log              # Training logs
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ️ Tech Stack
| Component | Tool/Library |
| :--- | :--- |
| Language | Python |
| DL Framework | TensorFlow + Keras |
| Image Handling | OpenCV, PIL |
| Data Handling | pandas, NumPy |
| Model Arch | CNN + BiLSTM + CTC Loss |
| API Deployment | FastAPI |
| Serving Env | Conda Virtual Env |

## Workflow
1.  **EDA** → understand dataset & image structure
2.  **Preprocessing**
    - Images cleaned & resized
    - Character map created & labels encoded
3.  **Model Architecture**
    - CNN extracts visual features
    - BiLSTM models the sequence
    - CTC allows unsegmented character prediction
4.  **Training & Validation**
    - CTC loss optimizes character-level accuracy
5.  **Inference**
    - New images → predicted medicine name
6.  **API Serving**
    - Expose prediction endpoint using FastAPI
    - Real-time image upload + response
