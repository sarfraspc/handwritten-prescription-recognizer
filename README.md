
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
│   ├── raw/                  
│   │   ├── Training/
│   │   ├── Validation/
│   │   └── Testing/
│   └── processed/           
│       ├── Training/
│       ├── Validation/
│       └── Testing/
│       └── char_map.json
│       └── train.csv / val.csv / test.csv
│
├── notebooks/
│   └── eda.ipynb             
│
├── scripts/
│   ├── clean_images.py       
│   ├── clean_val.py         
│   ├── clean_test.py        
│   └── train_model.py      
│
├── src/
│   ├── preprocessing/
│   │   ├── image_preprocessing.py    
│   │   ├── text_preprocessing.py  
│   │   └── __init__.py
│   ├── models/
│   │   └── crnn_ctc_model.py          
│   ├── training/
│   │   └── train_utils.py           
│   ├── inference/
│   │   └── predictor.py      
│   └── utils/
│       └── helpers.py                
│
├── tests/
│   └── test_pipeline.py       
│
├── app/
│   └── main.py               
│
├── logs/
│   └── train.log              
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
