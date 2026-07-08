# hybrid-ocr-llm-license-plate-recognition
Configuration Choose visibility * Choose who can see and commit to this repository  Add README READMEs can be used as longer descriptions. About READMEs On Add .gitignore .gitignore tells git which files not to track. About ignoring files Add license Licenses explain how others can use your code. About licenses
# Bridging Computer Vision and LLM Reasoning for Robust License Plate Recognition

## Overview

This project presents a hybrid Automatic License Plate Recognition (ALPR) framework that combines traditional Computer Vision (CV), Optical Character Recognition (OCR), and Large Language Model (LLM) reasoning to improve the recognition of Saudi vehicle license plates under challenging environmental conditions.

Unlike conventional ALPR systems that rely solely on pixel-level image processing, this framework integrates contextual reasoning from a vision-capable LLM to improve recognition accuracy, assess readability, and explain prediction failures.

The project was developed as part of a research study in the Artificial Intelligence Department at the University of Jeddah.

---

## Key Features

- Saudi license plate detection using Computer Vision
- OCR-based character extraction
- LLM-assisted reasoning and error correction
- Evaluation under multiple environmental distortions
- Performance comparison between CV/OCR and LLM approaches
- Interactive demo interface for real-time testing

---

## Methodology

The proposed framework consists of two complementary branches:

### Computer Vision (CV/OCR)

- License plate detection
- Image preprocessing
- Plate cropping and enlargement
- OCR text extraction
- Text normalization
- Performance evaluation against ground truth

### LLM Reasoning

The LLM branch analyzes the same image using structured prompting to:

- Read license plates
- Assess readability
- Estimate confidence
- Explain recognition failures
- Avoid unsafe guessing when the image quality is insufficient

The final framework compares the outputs of both branches to improve robustness under degraded image conditions.

---

## Environmental Distortions Evaluated

The system was evaluated under several challenging conditions, including:

- Gaussian Blur
- Motion Blur
- Low Resolution
- Low Light
- Dirt
- Partial Occlusion
- Rotation
- Perspective Tilt
- Over Exposure

These distortions simulate realistic conditions encountered in intelligent transportation systems.

---

## Technologies

- Python
- OpenCV
- Tesseract OCR
- Roboflow Inference API
- GPT-4o Vision
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Gradio

---

## Repository Structure

```
.
├── CV_Detection&OCR_Project.ipynb
├── cv_llm.ipynb
├── hybrid_cv_llm_pipeline.ipynb
├── results/
├── metadata_ground_truth_clean.csv
├── README.md
└── LICENSE
```

---

## Results

The hybrid framework demonstrates that combining Computer Vision with LLM reasoning significantly improves license plate recognition under degraded conditions.

The experiments show:

- Strong LLM performance on normal images
- Improved robustness under challenging environmental distortions
- Better explainability through reasoning-based analysis
- Identification of failure cases where both approaches struggle, particularly under severe blur and low-resolution images

---

## Applications

This work can support:

- Smart Cities
- Intelligent Transportation Systems
- Traffic Monitoring
- Parking Management
- Automated Access Control
- Vehicle Identification Systems

---

## Future Work

Future improvements include:

- Training a domain-specific OCR model for Saudi license plates
- Expanding the dataset
- Supporting multilingual license plates
- Real-time deployment
- Edge-device optimization
- Integration with additional Vision-Language Models

---

