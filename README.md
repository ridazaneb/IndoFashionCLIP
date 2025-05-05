# South Asian Fashion Trend Prediction with CLIP

S. Rida Zaneb  
Kenyon College  

---



This is a vision approach to South Asian fashion analysis by fine-tuningOpenAI’s **CLIPVisionModel** (image encoder only)  on the **IndoFashion** dataset—a balanced, 106 K-image corpus covering 15 categories of Indian ethnic garments. It uses CLIP’s vision encoder as a powerful feature extractor and fine-tunes it with a classification head to predict ethnic clothing categories.
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

- **Subset Used**: 500 images/class × 15 classes = **7,500 training images**
- **Metadata**: Each image labeled using newline-delimited JSON containing:
  - `image_path`
  - `class_label`
 
### Preprocessing

- Images resized to **224×224**
- Augmentations: `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation`
- Normalized to CLIP's input distribution
---

## 3. Model Architecture

### Vision Backbone
- `CLIPVisionModel` from HuggingFace
- Pretrained weights (`openai/clip-vit-base-patch32`)
- Outputs 512-dimensional image embeddings

### Classifier Head
- Linear layer: `nn.Linear(512, 15)`
- Softmax for prediction
- Cross-entropy loss

### Training Details

| Hyperparameter | Value |
|----------------|--------|
| Optimizer      | AdamW |
| Learning Rate  | 3e-5   |
| Epochs         | 15     |
| Batch Size     | 32     |
| Warmup Steps   | 10% of total |
| Early Stopping | 3-epoch patience |
| Device         | A100 GPU (Colab Pro) |

---

## 4. Evaluation Results

Model tested on a validation/test set with **500 samples/class** (7,500 total). Below are key metrics from the classification report:

### Performance Summary

| Class           | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| **Blouse**     | 0.96      | 0.88   | 0.92     |
| **Lehenga**    | 0.78      | 0.90   | 0.84     |
| **Mojaris (W)**| 0.75      | 0.87   | 0.80     |
| **Dupattas**   | 0.68      | 0.51   | 0.58     |
| **Dhoti Pants**| 0.63      | 0.42   | 0.51     |
| **Gowns**      | 0.59      | 0.55   | 0.57     |

- **Test Accuracy**: **74.1%**
- **Test Loss**: **0.9286**

> The model performs well on visually distinctive garments like *blouses*, but struggles with overlap-heavy categories like *gowns* and *dhoti pants*.

---

## 5. System Deployment

## 📊 Streamlit Dashboard

This app demonstrates the performance of our fine-tuned CLIP model on South Asian fashion images.

Features:
- Top-3 class predictions from fine-tuned model
- Prompt Engineering Playground (CLIP zero-shot matching)
- Interactive result plot carousel (F1 scores, precision-recall)


---

## 6. Challenges and Limitations

- **No text-image pairing** used (i.e., not a contrastive CLIP model)
- **Captioning** is skipped—faster, but may limit generalization
- **Visual ambiguity**: garments like *dupattas*, *gowns*, and *salwars* often misclassified due to stylistic similarity
- **Google Drive I/O Quota** is limited for a large training model like this causing significant delays

---

## 7. Future Directions

- **Incorporate Text**: Extend to full CLIP with caption alignment
- **Trend Forecasting**: Use temporal metadata and clustering
- **Multilingual Support**: Classify garments with regional caption inputs
- **Dataset Expansion**: Include more granular categories and regional styles

---

## 8. Equity Statement

This project supports racial and gender equity by highlighting underrepresented South Asian fashion. By adapting models like CLIP for cultural specificity, it helps de-Westernize AI fashion tools and promotes inclusion of traditional aesthetics in modern machine learning pipelines.
