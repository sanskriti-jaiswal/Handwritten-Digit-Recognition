## NumScan

### Overview
This project implements a **Handwritten Digit Recognition** system using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. The project consists of a **Python GUI application** built using `pygame` for digit input and a **Jupyter Notebook** for training and testing the model.

### Features
- **Handwritten digit input** using a Pygame-based GUI.
- **Deep learning model** trained with Keras on the MNIST dataset.
- **Real-time digit classification** displayed on the GUI.
- **Pre-trained model integration** for prediction.

### Project Structure
```
Handwritten-Digit-Recognition/
│-- app.py                # GUI for drawing and recognizing digits
│-- 1.ipynb               # Jupyter Notebook for training the model
│-- bestmodel.keras       # Pre-trained CNN model
│-- README.md             # Project documentation
```

### Requirements
Ensure you have the following dependencies installed:
```sh
pip install pygame keras tensorflow numpy opencv-python
```

### How to Run
1. **Train the Model (Optional):**
   - Open `1.ipynb` in Jupyter Notebook.
   - Run all cells to train the CNN model on the MNIST dataset.
   - Save the trained model as `bestmodel.keras`.

2. **Run the GUI Application:**
   ```sh
   python app.py
   ```
   - A window will open where you can draw a digit.
   - The model will recognize and display the predicted digit.

### Model Details
- The **CNN model** is trained using the MNIST dataset.
- Architecture consists of **convolutional, pooling, and fully connected layers**.
- Achieves **high accuracy** in handwritten digit classification.

### Tech Stack / Tools Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Pygame
- MNIST dataset

### Usage
- **Draw a digit** using the mouse in the GUI window.
- **Release the mouse** to allow the model to predict the digit.
- The **recognized digit** is displayed on the screen.
- Press **'N'** to clear the screen for a new input.

### How It Works
- User draws a digit on a Pygame canvas.
- The drawn image is captured and converted to grayscale using OpenCV.
- Image is thresholded, cropped, resized to 28x28, and normalized.
- The preprocessed image is reshaped and passed to a trained CNN model.
- The CNN predicts the digit based on the highest softmax probability.
- The predicted digit is displayed on the GUI in real-time.

### Acknowledgments
- **MNIST dataset** for handwritten digit recognition.
- **TensorFlow/Keras** for deep learning model implementation.
- **Pygame** for GUI development.

### Watch video
https://github.com/user-attachments/assets/9c1333cc-5dd5-4b87-b115-8ec6c99b2f15





