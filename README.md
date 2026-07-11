<div align="center">

#  Bridging Computer Vision and LLM Reasoning for Robust License Plate Recognition
### A Hybrid CV/OCR + Vision-LLM Framework for Saudi ALPR Under Real-World Degradation

*When Tesseract gives up on a blurry plate, does a vision-language model actually do better — or just guess more confidently?*

[![Status](https://img.shields.io/badge/status-completed-brightgreen)]()
[![Task](https://img.shields.io/badge/task-license%20plate%20recognition-blue)]()
[![CV Pipeline](https://img.shields.io/badge/CV-Roboflow%20%2B%20Tesseract-orange)]()
[![LLM](https://img.shields.io/badge/LLM-GPT--4o%20Vision-purple)]()
[![Demo](https://img.shields.io/badge/demo-Gradio-yellow)]()
[![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)]()

</div>

---

##  Overview

Automatic License Plate Recognition (ALPR) looks solved in a lab demo and falls apart in the real world. Sandstorms, motion blur, tilted cameras, blown-out midday sun, unlit desert highways at night — these are the actual operating conditions for Saudi Arabian traffic systems, and traditional pixel-based CV/OCR pipelines have no contextual reasoning to fall back on when characters are cracked, tilted, or partially obscured.

This project asks a direct question:

> **Can a modular hybrid pipeline that dynamically balances pixel-level CV/OCR detection with vision-capable LLM contextual reasoning offer a robust, explainable, and resource-efficient alternative to fully-trained specialized architectures under severe environmental distortion?**

To answer it, this project builds **two independent recognition branches** — a classic CV/OCR pipeline (Roboflow detector + Tesseract) and a vision-capable LLM reasoning branch (GPT-4o) — runs both against the *same* Saudi license plate images under nine distinct synthetic degradations, and does a rigorous, honest, side-by-side comparison. Not just "which one wins," but *where* each one breaks, and where **both** break — the boundary conditions that matter most for real deployment.

---

##  Research Context

Prior Saudi-specific ALPR work has largely gone in one of two directions: modular rule-based pipelines (edge detection + projection-based segmentation + OCR) that are interpretable but fragile under noise, or specialized deep-learning architectures purpose-built for Saudi's bilingual plate format:

| Prior Work | Approach |
|---|---|
| Antar et al. — *Automatic Number Plate Recognition of Saudi License Car Plates* | Classic rule-based pipeline: edge detection, adaptive thresholding, projection-based segmentation → OCR |
| Alwateer et al. — *XAI-SALPAD* | YOLOv8 joint plate/character detection + CNN classifier + LIME explainability |
| Alfehaid et al. | YOLOv5 vehicle-plate detector + specialized YOLO-based OCR module, bilingual mapping |
| Alhebshi et al. | Benchmarking of deep learning models specifically for Saudi arterial environments |
| Sarhan et al. (Egypt) | YOLOv8 + EasyOCR + custom CNN classifier |
| Sonnara et al. | Lightweight edge-device architectures balancing latency and power |
| AlDahoul et al. — *VehiclePaliGemma* | Multimodal VLM benchmarking (GPT-4o, Gemini, Claude) for plate recognition under degradation |

**What sets this project apart**: rather than training another specialized detector or deploying a heavyweight VLM as a replacement, this work treats CV/OCR and LLM reasoning as **two parallel, independently-evaluated branches** and systematically maps their *operational boundaries* — including the specific distortion types that defeat both simultaneously.

---

##  System Architecture

```
                        ┌───────────────────────────────────────────────┐
                        │   PHASE 1: Data Input & Preprocessing          │
                        │   46 pristine Saudi plates (Kaggle KSA dataset)│
                        │   → Image Distortion Module (OpenCV/NumPy)     │
                        │   → 9 distortion types × 46 = 414 test images  │
                        └───────────────────────┬───────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │            PHASE 2: Dual-Branch Processing        │
                        ├──────────────────────┬─────────────────────────┤
                        │   CV/OCR Branch        │   Vision-LLM Branch      │
                        │                       │                          │
                        │ Roboflow Inference API│ Image → Base64           │
                        │ (model rxg4e/11)      │        ↓                 │
                        │        ↓              │ GPT-4o Vision API        │
                        │ Plate Detect & Crop   │ (restrictive prompt —    │
                        │        ↓              │  no guessing, no meta-   │
                        │ Preprocess (5x resize)│  data inference)         │
                        │        ↓              │        ↓                 │
                        │ Tesseract OCR         │ Structured JSON output:  │
                        │ (PSM 6, alphanumeric  │ prediction, readability, │
                        │  whitelist)           │ confidence, failure      │
                        │        ↓              │ reason, safety flag      │
                        │ CV Prediction String  │ Structured Output        │
                        └──────────┬────────────┴────────────┬─────────────┘
                                   │                          │
                        ┌──────────┴──────────────────────────┴─────────────┐
                        │   PHASE 3: Evaluation & Comparison Logic           │
                        │   Ground Truth Benchmarking → Exact Match Acc.,    │
                        │   Character Similarity Score, Failure Mode Mapping │
                        └──────────────────────────┬──────────────────────┘
                                                     │
                        ┌────────────────────────────┴────────────────────────┐
                        │   PHASE 4: Collaborative Framework & Output           │
                        │   Groq Inference Layer (Llama 3.3 70B) → refined text │
                        │   → Final Prediction + Winning Path + Failure Reason  │
                        │     + Explainability Score                            │
                        └────────────────────────────────────────────────────┘
```

---

##  Dataset

| | |
|---|---|
| **Source** | [KSA License Plates Labeled dataset](https://www.kaggle.com/datasets/aamirhasankhan/ksa-license-plates-labled) (Kaggle) |
| **Baseline ("Normal") set** | 46 unique, manually verified Saudi vehicle images — 10% of the total evaluation dataset |
| **Ground truth** | Each of the 46 plates manually annotated for exact bilingual alphanumeric combination → certified Ground-Truth CSV |
| **Distorted set** | 46 images × 9 synthetic distortions = **414 degraded test instances** (90% of the total dataset) |
| **Matched distorted subset** | 90 images (10 per distortion type) used for the direct CV vs. LLM comparison |

### The 9 synthetic distortions (4 modalities)

| Modality | Distortions | Simulates |
|---|---|---|
| **Atmospheric & Motion** | Gaussian Blur, Motion Blur | Camera defocus / lens dust; high-speed transit |
| **Physical & Surface** | Dirt Filter, Partial Occlusion | Mud/dust buildup; obstructions or cracked plates |
| **Geometric & Perspective** | Rotation, Multi-axis Tilt | Misaligned camera mounts; acute viewing angles |
| **Illumination & Resolution** | Low Resolution, Low Light, Over-Exposure | Distant capture, unlit desert highways, harsh midday sun |

All distortions generated with **OpenCV** and **NumPy** via a custom augmentation module.

---

##  Branch 1 — CV/OCR Pipeline

1. **Detection**: Roboflow-hosted serverless inference API, model `license-plate-recognition-rxg4e/11` — first detection prediction used to crop the plate region
2. **Preprocessing**: crop enlarged **5× via cubic interpolation**
3. **OCR**: Tesseract, page segmentation mode **6**, alphanumeric whitelist
4. **Post-processing**: uppercase + alphanumeric-only cleaning rule
5. **Scoring**: exact match + character similarity against ground truth

**Alternatives considered but not used in the final pipeline**: Canny edge detection, Sobel filtering, EasyOCR, custom CNN-based recognition — reviewed during the analysis phase but Roboflow + Tesseract was the final implementation.

---

##  Branch 2 — Vision-LLM Reasoning Pipeline

- **Model**: vision-capable **GPT-4o**
- **Input**: images encoded as base64 data URLs
- **Prompt design (critical)**: a strict, standardized engineering prompt that:
  - Constrains the model to analyze **only the current visual input** — no inference from filenames or metadata
  - **Explicitly prohibits guessing** hidden/unclear characters or completing text based on expected plate formats
  - Requires a **structured JSON output**: plate prediction, readability, confidence score, failure reason, and prediction safety
  - Returns `*unreadable*` instead of a low-confidence guess when the plate genuinely isn't legible

This prompt design is the difference between "an LLM that hallucinates plausible-looking plates" and "an LLM that tells you honestly when it can't read something" — which matters enormously for a system where a wrong-but-confident answer is worse than no answer.

---

##  Results

### CV/OCR Performance

| Condition | Exact-Match Accuracy | Avg. Character Similarity | Avg. Detection Confidence |
|---|---|---|---|
| Normal (46 images) | **8.70%** (4/46) | 57.22% | 85.31 |
| Full distorted set (414 images) | **2.17%** (9/414) | — | — |
| Matched distorted subset (90 images) | **4.44%** (4/90) | — | — |

**Full CV/OCR distorted performance by degradation type:**

| Distortion | Valid | Correct | Exact Acc. | Char Sim. | Detect Conf. |
|---|---|---|---|---|---|
| Dirty | 46 | 2 | 4.35% | 56.72% | 85.37 |
| Gaussian Blur | 46 | 0 | 0.00% | 5.37% | 74.52 |
| Low Light | 46 | 4 | 8.70% | 54.80% | 85.23 |
| Low Res | 46 | 0 | 0.00% | 3.21% | 74.96 |
| Motion Blur | 46 | 0 | 0.00% | 2.87% | 68.48 |
| Occlusion | 46 | 2 | 4.35% | 54.25% | 84.92 |
| Over Exposed | 46 | 1 | 2.17% | 44.80% | 84.27 |
| Rotation | 46 | 0 | 0.00% | 19.36% | 83.60 |
| Tilted | 46 | 0 | 0.00% | 15.45% | 83.31 |

 **A crucial pattern**: detection confidence stays high (74–85) across nearly every category, even when exact-match accuracy is 0%. **Successful plate *detection* does not guarantee correct OCR *recognition*.** The bottleneck is character extraction from the crop, not localization.

### LLM Performance

| Condition | Exact-Match Accuracy | Avg. Confidence |
|---|---|---|
| Normal (46 images) | **91.30%** (42/46) | 88.13 |
| Distorted matched subset (90 images) | **66.67%** (60/90) | 66.51 |

**LLM distorted performance by degradation type:**

| Distortion | Valid | Correct | Exact Acc. | Char Sim. | LLM Conf. |
|---|---|---|---|---|---|
| Dirty | 10 | 10 | **100.00%** | 100.00% | 90.30 |
| Gaussian Blur | 10 | 1 | 10.00% | 25.24% | 20.10 |
| Low Light | 10 | 10 | **100.00%** | 100.00% | 85.60 |
| Low Res | 10 | 3 | 30.00% | 36.67% | 31.00 |
| Motion Blur | 10 | 1 | 10.00% | 25.84% | 19.70 |
| Occlusion | 10 | 9 | 90.00% | 99.09% | 87.50 |
| Over Exposed | 10 | 9 | 90.00% | 97.27% | 89.10 |
| Rotation | 10 | 9 | 90.00% | 98.57% | 91.10 |
| Tilted | 10 | 8 | 80.00% | 97.14% | 84.20 |

 Notice the confidence scores **track actual performance honestly** — confidence craters to ~20 on Gaussian/Motion Blur (where accuracy is also worst) rather than staying artificially high. The prompt's "don't guess" constraint appears to be working as intended.

### Matched CV/OCR vs. LLM Comparison (90-image subset)

| Distortion | CV Acc. | LLM Acc. | Reasoning Quality Proxy | Both-Failed Rate |
|---|---|---|---|---|
| Dirty | 10.00% | 100.00% | 1.10 / 2 | 0.00% |
| **Gaussian Blur** | 0.00% | 10.00% | 2.00 / 2 | **90.00%** |
| Low Light | 10.00% | 100.00% | 1.30 / 2 | 0.00% |
| Low Res | 0.00% | 30.00% | 1.70 / 2 | 70.00% |
| **Motion Blur** | 0.00% | 10.00% | 1.40 / 2 | **90.00%** |
| Occlusion | 10.00% | 90.00% | 1.50 / 2 | 10.00% |
| Over Exposed | 10.00% | 90.00% | 1.80 / 2 | 10.00% |
| Rotation | 0.00% | 90.00% | 1.10 / 2 | 10.00% |
| Tilted | 0.00% | 80.00% | 1.70 / 2 | 20.00% |

###  Both-Failed Analysis — the real headline finding

**Gaussian blur and motion blur are the hard boundary for *both* systems** — each producing a **90.00% joint failure rate**. Low resolution follows at 70.00%. These three categories share a common mechanism: they destroy character-edge information at the pixel level, which removes the visual evidence both OCR *and* LLM reasoning depend on — no amount of contextual reasoning can read a character whose edges no longer exist in the image.

By contrast, dirt, low-light, occlusion, over-exposure, rotation, and tilt were far less destructive for the LLM specifically — because in those cases, enough character *structure* remains visible for contextual reasoning to fill the gap, even where raw OCR fails outright.

---

##  Demo System

An interactive **Gradio** demo runs the full hybrid pipeline end-to-end:

1. User uploads a single Saudi license-plate image
2. Image is processed through **both** branches simultaneously (CV/OCR + GPT-4o vision)
3. Interface displays: predicted plate text from each branch, confidence scores, readability assessment, safety status, and the **final hybrid decision**
4. A short natural-language explanation describes *why* a particular branch's output was selected as final

**Safety-first design**: if the LLM branch determines a plate is not sufficiently readable, it returns `*unreadable*` rather than emitting an uncertain guess — reducing unsafe/misleading outputs in low-quality conditions.

A **Groq-hosted Llama 3.3 70B** inference layer sits downstream as a collaborative decision module, generating the refined final explanation, winning-path determination, and an explainability score.

---

##  Discussion

- The CV/OCR branch's core weakness isn't detection — it's character-level extraction. High detection confidence coexisting with 0% exact-match accuracy (see Gaussian Blur, Motion Blur, Low Res, Rotation, Tilted rows) makes this explicit: **Tesseract is highly sensitive to blur, low resolution, small character size, and partial visibility**, and because exact-match scoring is strict, a single wrong or missing character fails the whole prediction.
- The LLM branch's advantage was strongest precisely where **some visual structure survived** — dirt, lighting shifts, occlusion, tilt, rotation — situations where contextual reasoning can bridge gaps that break character-segmentation-based OCR outright.
- Both approaches share a hard physical limit: **when character-edge information is genuinely destroyed** (Gaussian/motion blur, low resolution), no amount of contextual reasoning compensates. This is a meaningful, generalizable finding for real-world ALPR deployment — it tells you exactly which physical/optical conditions (camera stabilization, minimum resolution, shutter speed) actually need to be solved at the *hardware* level rather than the *software* level.

---

##  Limitations & Challenges

- Some normal-image samples lacked reliable ground-truth labels and were excluded from analysis.
- The CV/OCR branch used an **off-the-shelf Roboflow detector**, not one specifically trained for Saudi plates — performance is dependent on preprocessing quality and general-purpose detection accuracy.
- LLM evaluation introduces **API cost, latency, and model-variability** concerns; due to time/cost constraints, the full 414-image distorted set was *not* evaluated with the LLM — only the 90-image matched subset was.
- Slight differences were observed between offline evaluation and live API outputs.
- **Reasoning quality was scored via a rubric-based proxy**, not independent human annotation.
- Quantitative evaluation focused on **normalized Latin alphanumeric strings**, not full bilingual (Arabic + Latin) Saudi plate transcription.

---

##  Tech Stack

| Component | Technology |
|---|---|
| Plate detection | Roboflow Inference API (`license-plate-recognition-rxg4e/11`) |
| OCR | Tesseract (`pytesseract`), PSM 6, alphanumeric whitelist |
| Image augmentation | OpenCV, NumPy (9 custom distortion simulators) |
| Vision-LLM reasoning | GPT-4o (vision API), structured JSON output |
| Collaborative refinement layer | Groq — Llama 3.3 70B |
| Demo interface | Gradio |
| Data handling | pandas, tqdm |
| Environment | Jupyter Notebook |

---

##  Repository Structure

```
.
├── cv_llm.ipynb                                          # Core hybrid CV+LLM comparison pipeline
├── hybrid_cv_llm_pipeline (2).ipynb                       # End-to-end hybrid pipeline + Gradio demo
├── CV_Detection&OCR_Project.ipynb                         # Standalone CV/OCR branch (Roboflow + Tesseract)
│
├── metadata_ground_truth_clean.csv                        # Certified ground-truth CSV (46 normal plates)
├── cv_llm4_ground_truth_10_each (1).xlsx                  # Ground truth for the 90-image matched subset
├── cv_ocr_normal_results.csv                               # CV/OCR results — normal (undistorted) images
├── cv_ocr_distorted_results.csv                            # CV/OCR results — full 414-image distorted set
├── cv_ocr_10distorted_results_matched_with_distortions.csv # CV/OCR results — 90-image matched subset
│
├── LICENSE
├── .gitignore
└── README.md
```

---

##  Future Work

- Evaluate the LLM branch across the **full** 414-image distorted set (not just the 90-image matched subset), cost/latency permitting
- Extend quantitative scoring to full **bilingual Arabic + Latin** plate transcription rather than normalized Latin-only strings
- Replace the rubric-based reasoning-quality proxy with independent human annotation
- Fine-tune or adapt the detection model specifically on Saudi plate geometry rather than relying on a general-purpose Roboflow model
- Investigate targeted mitigation for the Gaussian-blur/motion-blur/low-resolution failure boundary — e.g., deblurring pre-processing or super-resolution upscaling before either branch runs

---

##  References

1. M. Alwateer, K. O. Aljuhani, A. Shaqrah, R. ElAgamy, G. Elmarhomy, and E.-S. Atlam, "XAI-SALPAD: Explainable deep learning techniques for Saudi Arabia license plate automatic detection," *Alexandria Engineering Journal*, 2024. https://doi.org/10.1016/j.aej.2024.09.057
2. R. K. Alfehaid, S. M. Almutairi, and Q. E. U. Haq, "A Deep Learning Based Approach for License Plate Recognition," *Journal of Information Security and Cybercrimes Research*, 8(2), 167–177, 2025. https://doi.org/10.26735/TKIP7914
3. R. M. Alhebshi, A. Almakky, L. W. Aljahdali, and R. B. Hawsawi, "Deep Learning Approaches for License Plate Detection in Saudi Arabia," *Journal of King Abdulaziz University: Computing and Information Technology Sciences*, 14(2), 2025. https://doi.org/10.64064/1658-6336.1012
4. R. Antar, S. Alghamdi, J. Alotaibi, and M. Alghamdi, "Automatic Number Plate Recognition of Saudi License Car Plates," *Engineering, Technology & Applied Science Research*, 12(2), 8266–8272, 2022.
5. F. Sonnara, H. Chihaoui, and F. Filali, "Efficient real-time license plate recognition using deep learning on edge devices," *Journal of Real-Time Image Processing*, 22, art. 159, 2025. https://doi.org/10.1007/s11554-025-01738-3
6. A. Sarhan et al., "Egyptian car plate recognition based on YOLOv8, Easy-OCR, and CNN," *Journal of Electrical Systems and Information Technology*, 11, art. 32, 2024. https://doi.org/10.1186/s43067-024-00156-y
7. N. AlDahoul et al., "Multitasking vision language models for vehicle plate recognition with VehiclePaliGemma," *Scientific Reports*, 2025. https://doi.org/10.1038/s41598-025-10774-9
8. A. H. Khan, "KSA License Plates Labeled Dataset," Kaggle, 2024. https://www.kaggle.com/datasets/aamirhasankhan/ksa-license-plates-labled

---

##  Authors

**University of Jeddah — College of Computer Science and Engineering — Artificial Intelligence Department**
Supervisor: Dr. Mashael Buqari · May 2026

- Atheer Alsulami 
- Noor Bamuhair 
- Zahrah Saleh 

---

<div align="center">

*Detection confidence of 85 and exact-match accuracy of 0% — the gap between "found a plate" and "read a plate" is the whole project.*

</div>
