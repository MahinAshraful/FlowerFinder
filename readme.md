# Handwritten Digit Recognition

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from images. The model is trained using the MNIST dataset and can predict digits from new images.

## Project Structure


- `digits/`: Directory containing images of handwritten digits to be predicted.
- `handwritten.keras`: Pre-trained Keras model for digit recognition.
- `main.py`: Main script to load the model and predict digits from images.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/handwritten-digit-recognition.git
    cd handwritten-digit-recognition
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow opencv-python numpy matplotlib
    ```

## Usage

1. Place your digit images in the `digits/` directory. The images should be named as `digit1.png`, `digit2.png`, etc.

2. Run the `main.py` script:
    ```sh
    python main.py
    ```

3. The script will load the pre-trained model, read the images, and predict the digits. The predicted digit will be printed in the console, and the image will be displayed.

## Example

The number is probably: 7


## Model Training

The model is trained using the MNIST dataset. The training code is commented out in `main.py`:

```python
# mnist = tf.keras.datasets.mnist # 28x28 images of hand-written digits 0-9
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1) 
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=100)

# model.save('handwritten.keras')

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
TensorFlow
OpenCV
NumPy
Matplotlib
MNIST Dataset
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.