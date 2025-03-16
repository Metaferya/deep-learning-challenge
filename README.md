# Alphabet Soup Charity Deep Learning Model

## Overview
This project builds a deep learning model to predict the success of non-profit organizations applying for funding from Alphabet Soup. The model processes application data, applies feature engineering, trains a deep neural network, and optimizes it for better accuracy.

## Dataset
- **Source:** Provided cloud URL
- **Target Variable:** `IS_SUCCESSFUL`
- **Feature Variables:**
  - `APPLICATION_TYPE`
  - `AFFILIATION`
  - `CLASSIFICATION`
  - `USE_CASE`
  - `ORGANIZATION`
  - `INCOME_AMT`
  - `SPECIAL_CONSIDERATIONS`
  - `ASK_AMT`
  - `IS_SUCCESSFUL`
- **Dropped Columns:**
  - `EIN` (Not relevant)
  - `NAME` (Not useful for prediction)

## Preprocessing Steps
- **Categorical Encoding:** Used `pd.get_dummies()` to convert categorical variables.
- **Feature Engineering:** Grouped rare categories under "Other" for `APPLICATION_TYPE` and `CLASSIFICATION`.
- **Feature Scaling:** Applied `StandardScaler()` to normalize numerical features.

## Model Architecture
- **Input Layer:** Number of features from preprocessed data.
- **Hidden Layers:**
  - **Layer 1:** 128 neurons, LeakyReLU activation
  - **Layer 2:** 64 neurons, tanh activation
  - **Layer 3:** 32 neurons, LeakyReLU activation
  - **Layer 4:** 16 neurons, tanh activation
- **Output Layer:** 1 neuron, sigmoid activation (binary classification)

## Training and Optimization
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Initially RMSprop, but changed to Adam for better performance
- **Epochs:** 100
- **Batch Size:** 32
- **Callbacks:** ModelCheckpoint saves model weights every 5 epochs

## Model Evaluation
- **Loss:** `{model_loss}`
- **Accuracy:** `{model_accuracy}` (~72%, below target 75%)

## Model Accuracy Plot
Below is the accuracy plot of training vs. validation accuracy over epochs:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
```

## Optimization Attempts
- Increased neurons in hidden layers
- Added a fourth hidden layer
- Experimented with different activation functions (LeakyReLU, tanh)
- Extended training epochs from 50 to 100
- Implemented ModelCheckpoint to save progress

## Recommendations
- Try the **Adam optimizer** instead of RMSprop
- Add **EarlyStopping callback** to prevent overfitting
- Reduce the number of hidden layers if needed
- Consider alternative models such as:
  - **Random Forest** (Good for structured categorical data)
  - **XGBoost** (Often better than deep learning in tabular datasets)
  - **SVM** (Effective for binary classification)

## Next Steps
1. Further optimize the deep learning model to exceed 75% accuracy.
2. Train and evaluate the model in Google Colab (`AlphabetSoupCharity_Optimization.ipynb`).
3. Push the final model (`AlphabetSoupCharity_Optimization.h5`) to GitHub.

---

