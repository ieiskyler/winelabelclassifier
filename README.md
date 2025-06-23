# Wine Label Classifier

This project is an image classification pipeline for wine label images using deep learning. It includes a baseline CNN and transfer learning approaches (MobileNetV2 and MobileNetV3Small) to classify wine types from label images.

---

## Features

- **Data Preparation:** Loads metadata and images, encodes labels, and splits data into training and validation sets.
- **Data Augmentation:** Uses Keras `ImageDataGenerator` for robust training.
- **Baseline CNN:** Simple convolutional neural network for comparison.
- **Transfer Learning:** Fine-tuned MobileNetV2 and MobileNetV3Small models for improved accuracy.
- **Visualization:** Plots training curves and prediction results.
- **Evaluation:** Reports validation loss and accuracy for all models.

---

## Project Structure

```
.
├── classifier.ipynb      # Main Jupyter notebook with all code
├── Dataset/
│   └── last/
│       ├── XWines_Test_100_wines.csv
│       └── Xwines_Test_100_labels/
│           └── *.jpeg
└── README.md
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas, numpy, matplotlib, scikit-learn

Install dependencies:
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

---

## Usage

1. **Prepare Data:**  
   Place your CSV and image files in the `Dataset/last/` directory as shown above.

2. **Run the Notebook:**  
   Open `classifier.ipynb` in VS Code or Jupyter and run all cells.

3. **Training:**  
   - The notebook will train a baseline CNN and two transfer learning models.
   - Early stopping and model checkpoints are used for best results.

4. **Evaluation & Visualization:**  
   - Validation accuracy and loss are printed for each model.
   - Training curves and sample predictions are visualized.

---

## Model Architectures

### Baseline CNN
- 2–3 Conv2D layers with MaxPooling and Dropout
- Dense layers for classification

### Transfer Learning
- **MobileNetV2** and **MobileNetV3Small** as feature extractors
- Custom dense layers on top
- Fine-tuning of top layers for best performance

---

## Results

- **Baseline CNN:** Provides a simple benchmark.
- **MobileNetV2/V3:** Achieve higher accuracy, especially with small datasets.

---

## Tips

- For best results, use transfer learning models.
- If validation accuracy is low, try increasing model depth, using more data, or tuning augmentation.
- You can adjust the number of trainable layers in MobileNet for fine-tuning.

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [scikit-learn](https://scikit-learn.org/)