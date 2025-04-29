# South Asian Fashion Trend Prediction with CLIP

S. Rida Zaneb  
Kenyon College  

---



This is a multimodal approach to South Asian fashion analysis by fine-tuning OpenAI’s CLIP model on the **IndoFashion** dataset—a balanced, 106 K-image corpus covering 15 categories of Indian ethnic garments. Beyond classification, we perform embedding-based clustering to uncover emerging style trends. Our pipeline also includes a Streamlit dashboard for interactive exploration and prediction.

---

## 1. Background & Motivation

- **Underrepresentation:** Most fashion-AI research and datasets focus on Western attire.  
- **Cultural Nuance:** South Asian garments (e.g., sarees, kurtas, lehengas) exhibit unique draping, patterns, and contextual usage that generic models misclassify.  
- **Market Impact:** South Asia is one of the world’s largest apparel markets; culturally aware AI tools can democratize design insights for regional designers and retailers.

---

## 2. Dataset: IndoFashion

| Split       | #Images | Description                                     |
|-------------|--------:|-------------------------------------------------|
| Training    |  91,166 | 15 class-balanced categories                    |
| Validation  |   7,500 | Held-out for hyperparameter tuning              |
| Test        |   7,500 | Final held-out for reporting results            |

- **Sources:** Scraped from Amazon, Flipkart, Myntra, Google Image Search.  
- **Annotations:**  
  - `class_label` (e.g. `saree`, `lehenga`, `kurta`)  
  - `product_title`, `brand`, `color` (optional)  
  - **Synthetic captions** generated via regex rules (e.g. `"a red saree"`) for CLIP alignment.

---

## 3. Methodology

### 3.1 CLIP Fine-Tuning

1. **Model Choice:** `openai/clip-vit-base-patch32`  
2. **Contrastive Objective:** maximize cosine similarity between image and its synthetic caption  
3. **Training Details:**  
   - Optimizer: AdamW (lr = 5×10⁻⁶)  
   - Batch size: 64 images/texts  
   - Epochs: 4 (adjustable)  
   - Hardware: A100 GPU (Colab Pro+), 24 GB RAM

### 3.2 Embedding-Based Trend Clustering

1. **Feature Extraction:** freeze fine-tuned CLIP, extract 512-dim image embeddings  
2. **Dimensionality Reduction:** PCA to 50 components  
3. **Clustering:** K-means (k = 12) to group visually and semantically similar garments  
4. **Temporal Analysis (Future):** correlate cluster frequencies with date metadata for seasonality insights

---

## 4. Experimental Setup

- **Environment:**  
  - Python 3.10, PyTorch 2.0, Transformers 4.30  
  - Dependencies: see [requirements.txt](./requirements.txt)  
- **Data Storage:**  
  - Raw images & JSON in Google Drive (2.7 GB)  
  - Processed data — CSV splits + resized images (224×224)  
- **Compute:**  
  - Colab Pro (T4/GPU) or local A100

---

## 5. Evaluation Metrics

| Metric         | Definition                                      | Purpose                                  |
|---------------:|-------------------------------------------------|------------------------------------------|
| Accuracy       | (TP+TN)/Total                                   | Overall classification performance       |
| Precision/Recall/F1 (per class) | Standard definitions                     | Address class imbalance & error types    |
| Recall@K (CLIP)     | Fraction of correct caption in top-K matches | Assess raw text–image alignment quality  |

---

## 6. Results Summary

| Split       | Accuracy | Avg. F1  |
|------------:|---------:|---------:|
| Validation  |   𝟴𝟭.𝟲 % |  0.81    |
| Test        |   𝟴𝟮.𝟯 % |  0.82    |

> **Note:** raw cosine-softmax confidences hover near uniform; we recommend adding a small MLP head or temperature scaling for better calibration.

---

## 7. Streamlit Dashboard

- **File:** `scripts/app_dashboard.py`  
- **Features:**  
  1. Upload an image → predict top-3 garment classes with confidences  
  2. “Similar styles” gallery via clustering  
- **Launch:**  
  ```bash
  streamlit run scripts/app_dashboard.py
