# 🎨 Fashion Product Attribute Prediction

This project aims to classify fashion products based on their **style**, **category**, and **base colour** using structured metadata and image embeddings. We used deep learning models (MLPs) on top of embeddings generated via ResNet50.

---

## 📂 Dataset

We used the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset from Kaggle.

**Dataset Overview:**

- Total Entries: 44,424
- Features:
  - `id` (integer)
  - `gender` (object)
  - `masterCategory` (object)
  - `subCategory` (object)
  - `articleType` (object)
  - `baseColour` (object)
  - `season` (object)
  - `year` (float)
  - `usage` (object)
  - `productDisplayName` (object)

> **Note:** The dataset exhibited significant class imbalance. We performed data cleaning and focused on relevant data subsets for each prediction task to achieve more balanced and meaningful results.

---

## 🎯 Prediction Objectives

We trained **three separate MLP models** to predict:

1. **Style**: `casual`, `formal`, `sportswear`
2. **Category**: `top`, `bottom`, `footwear`
3. **Base Colour**: Top 10 grouped common colours

---

## 🧪 Preprocessing & Feature Engineering

- Data was **cleaned** to remove missing, duplicate, and potentially unreliable entries.
- **Label Encoding** and **Binary Encoding** were employed for categorical variables.
- For generating feature vectors, we used **ResNet50** (pre-trained on ImageNet) to extract **2048-dimensional** embeddings from the product images.
- **Base Colour** labels were grouped into the **10 most frequent shades** to improve model generalization.

---

## 🧠 Model Architecture

### 🔷 Style and Category Classifier

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

def create_mlp_model(input_dim, output_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

### 🔷 Base Colour Classifier
```Python

def create_mlp_model_for_color(input_dim, output_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```
### 📊 Model Performance
#### ✔️ Accuracy Overview
| Task        | Accuracy |
|-------------|----------|
| Style       | 87.67%   |
| Category    | 99.78%   |
| Base Colour | 72.81%   |


### 📈 Classification Reports

#### 🔷 Style Prediction

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| casual     | 0.96      | 0.88   | 0.92     | 3515    |
| formal     | 0.72      | 0.93   | 0.81     | 363     |
| sportswear | 0.66      | 0.85   | 0.74     | 712     |
| Accuracy   |           |        | 0.88     | 4590    |



### 🧥 Category Prediction

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| bottom   | 0.99      | 0.99   | 0.99     | 506     |
| footwear | 1.00      | 1.00   | 1.00     | 1464    |
| top      | 1.00      | 1.00   | 1.00     | 2620    |
| Accuracy |           |        | 1.00     | 4590    |



#### 🎨 Base Colour Prediction

| Colour   | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Black    | 0.81      | 0.78   | 0.80     | 949     |
| Blue     | 0.86      | 0.69   | 0.77     | 900     |
| Brown    | 0.67      | 0.66   | 0.66     | 377     |
| Green    | 0.69      | 0.76   | 0.73     | 291     |
| Grey     | 0.52      | 0.61   | 0.56     | 435     |
| Multi    | 0.19      | 0.35   | 0.25     | 20      |
| Orange   | 0.49      | 0.67   | 0.57     | 48      |
| Pink     | 0.68      | 0.73   | 0.70     | 186     |
| Purple   | 0.58      | 0.73   | 0.65     | 177     |
| Red      | 0.75      | 0.82   | 0.78     | 314     |
| White    | 0.79      | 0.77   | 0.78     | 752     |
| Yellow   | 0.63      | 0.77   | 0.70     | 141     |
| Accuracy |           |        | 0.73     | 4590    |



### 🙌 Acknowledgements

Fashion Product Images (Small) by [Param Aggarwal on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).
