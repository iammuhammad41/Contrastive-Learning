# Contrastive Learning for Cassava Leaf Disease Classification

This project implements a supervised contrastive learning pipeline on top of an EfficientNet backbone to classify cassava leaf diseases. By first pre‑training an encoder with a contrastive loss and then training a classifier head, we learn robust representations and achieve high classification accuracy.



## ⚙️ Project Structure

```
contrastive-learning-cassava/
├── notebook.ipynb               # Main Colab/Kaggle notebook with full pipeline
├── scripts/                     # (Optional) Python scripts extracted from notebook
│   ├── data_utils.py            # TFRecord parsing, augmentations, dataset prep
│   ├── model_utils.py           # Encoder, projection head, classifier definitions
│   ├── train_encoder.py         # Script to pre‑train encoder with SupCon loss
│   └── train_classifier.py      # Script to train classifier head
├── outputs/                     # Saved weights, logs, and visualizations
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and instructions
```



## 🔧 Dependencies

* Python 3.7+
* TensorFlow 2.x
* tensorflow-addons
* efficientnet-tfkeras
* KaggleDatasets
* scikit-learn
* matplotlib, seaborn
* pandas, numpy, tqdm

Install with:

```bash
pip install tensorflow tensorflow-addons efficientnet kaggle-datasets scikit-learn matplotlib seaborn pandas numpy tqdm
```



## 🖥️ Environment

* Supports TPU (via `TPUClusterResolver`) or GPU
* Mixed precision (`bfloat16`) and XLA JIT enabled
* Batch size and learning rate are scaled by TPU replicas



## 🚀 Usage

1. **Clone repository** and open `notebook.ipynb` in Colab or Kaggle.
2. **Authenticate** KaggleDatasets to access pre‑prepared TFRecords for the Cassava Leaf Disease dataset.
3. **Run all cells** sequentially:

   * Hardware detection and strategy setup
   * Dataset paths and TFRecord file listing
   * Data augmentation and parsing functions
   * Supervised contrastive encoder pre‑training (`EPOCHS_SCL`)
   * Classifier training with frozen encoder (`EPOCHS`)
   * K‑fold cross validation (OOF embeddings, metrics)
   * t‑SNE visualization of learned representations
   * Classification reports and confusion matrices

Checkpoints and logs are saved in `outputs/` by fold.



## 🧠 Model Components

* **Encoder**: EfficientNetB3 (noisy‑student weights), output = pooled feature vector
* **Projection Head**: Dense(128, ReLU) for contrastive training
* **Supervised Contrastive Loss**: encourages same‑class samples to cluster
* **Classifier Head**: Dense layers + softmax for final 5‑class prediction



## 📊 Evaluation

* **Metrics**: Sparse Categorical Accuracy, loss during training; classification report on OOF splits
* **Visualizations**:

  * Learning curves for encoder and classifier
  * t‑SNE plots of embeddings pre‑ and post‑contrastive training
  * Confusion matrices (normalized)
  * Sample predictions plotted on images



## 🔧 Configuration

Edit constants at the top of the notebook or scripts:

```python
BATCH_SIZE = 64 * REPLICAS
LEARNING_RATE = 3e-5 * REPLICAS
EPOCHS_SCL = 15      # contrastive pre‑training
EPOCHS = 10          # classifier training
HEIGHT, WIDTH = 512, 512
N_CLASSES = 5
N_FOLDS = 5
FOLDS_USED = 1       # number of folds to run
```

Adjust data augmentation parameters in `data_augment()` and learning rate schedule in `cosine_schedule_with_warmup()` as needed.
