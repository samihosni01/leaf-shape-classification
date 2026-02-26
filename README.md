# Leaf Classification Challenge: ResNet vs. Gradient Boosting 🪴🍃

Hey, this is my solution for the [Kaggle Leaf Classification Challenge](https://www.kaggle.com/competitions/leaf-classification). The task was to classify 990 leaf images into 99 different tree species 

I built two approaches to compare: a **Deep Learning model with ResNet** and a **classical ML model with Gradient Boosting + HOG features**.

## 📊 About the Dataset
* **Training:** 990 images from 99 species (about 10 images per class)
* **Test:** 594 images to predict
* Images are stored in `/images/` with IDs in `train.csv` and `test.csv`
* **Challenge:** Very small dataset (only ~10 samples per class) and lots of very similar-looking leaves!

*Note: Images are huge so I didn't include them here. Download from Kaggle!*

## 🧠 My Approach

### 1. Deep Learning: ResNet-18 (Transfer Learning)
I used a pre-trained **ResNet-18** (ImageNet weights) because:
- Vision Transformers need way more data (100k+ images) to avoid overfitting
- With only 990 images, ResNet is perfect - pre-trained on 1.2M images

**Key tricks I used:**
- Froze the backbone, only trained the final layer
- Added **Dropout (20%)** and **L2 regularization** to fight overfitting
- Data augmentation: RandomCrop + HorizontalFlip
- Trained 25 epochs, saved the best model (92.62% validation accuracy!)

### 2. Classical ML: XGBoost + HOG Features
For the ML approach:
- **HOG features** (scans WHOLE image for edges/textures - perfect for leaf shape/margin/veins) vs SIFT (only detects keypoints/spots like curves - unsuitable for uniform leaf surfaces)
- **XGBoost** (250 estimators, max_depth=50) - each tree fixes the previous one's mistakes
- Train/val split: 85/15
- Got **71.14% validation accuracy**

## 🛠️ Tech I Used
* **Deep Learning:** PyTorch, Torchvision (ResNet-18), DataLoader
* **Machine Learning:** XGBoost, Scikit-Learn, Scikit-Image (HOG)
* **Data:** Pandas, PIL, NumPy
* **Hardware:** GPU (CUDA) for faster training

## 📈 Results
**Validation Accuracy:**
* **ResNet-18:** **92.62%** 🏆 (way better, as expected with transfer learning)
* **XGBoost + HOG:** **71.14%** 

**Why ResNet crushed it:** With small data, pre-trained CNNs like ResNet are unbeatable. They already "know" leaf-like shapes from ImageNet. XGBoost did okay but struggled with the fine differences between similar species.

Both models generated Kaggle submissions:
- `submission_ResnetCNN.csv` (full probabilities for all 99 classes)
- `submission_GB_HOG.csv` (single predictions)

## 💡 What I Learned
- **Small data = Transfer Learning wins every time.** ViT would have overfit badly here.
- HOG is solid for texture-heavy images like leaves.
- XGBoost > Random Forest because it learns sequentially from mistakes.
- Always use Dropout + L2 + augmentation for image classification!

