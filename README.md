# Eye Disease Classification using CNN

This project is a deep learning-based image classification model designed to identify different types of eye diseases from images. The model is built using TensorFlow and Keras and trained on an image dataset of eye conditions.

## Features
- Convolutional Neural Network (CNN) model implementation
- Uses TensorFlow and Keras for deep learning
- Image preprocessing with ImageDataGenerator
- Supports categorical classification for multiple eye disease categories
- Model evaluation and visualization of predictions

## Dataset
The dataset used in this project is located at:
```
/kaggle/input/eye-dataset/Eye dataset
```
The dataset contains multiple classes of eye diseases, and images are stored in directories corresponding to their labels.

## Installation
To use this project, ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib
```

## Model Architecture
The CNN model consists of:
- **Conv2D Layers:** Feature extraction through convolutional layers
- **MaxPooling2D Layers:** Reducing spatial dimensions to prevent overfitting
- **Flatten Layer:** Converts feature maps into a single-dimensional vector
- **Dense Layers:** Fully connected layers for classification

## Training the Model
The model is compiled with categorical cross-entropy loss and Adam optimizer. The training is performed using `fit()` with a validation set.
```python
model.fit(train_generator, validation_data=validation_generator, epochs=10)
```

## Model Evaluation
After training, the model can be evaluated using:
```python
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")
```

## Prediction Function
The `gercek_deger` function allows prediction on new images:
```python
def gercek_deger(image_path, model, class_indices):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class]
    
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis("off")
    plt.show()
```

## Saving and Loading the Model
The trained model is saved as `image_classifier.h5` and can be loaded using:
```python
from tensorflow.keras.models import load_model
model = load_model("image_classifier.h5")
```

## Future Improvements
- Implement transfer learning with pre-trained models (e.g., VGG16, ResNet)
- Expand dataset with additional eye disease images
- Develop a web or mobile application for real-time diagnosis

## License
This project is open-source and available under the MIT License.

## Contact
For questions or contributions, please open an issue on GitHub.

