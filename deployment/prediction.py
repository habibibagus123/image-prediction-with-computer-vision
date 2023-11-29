import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('model.h5')  # Load your .h5 model file

# Define class names for interpretation
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def preprocess_image(image):
    # Resize the image to the required input size (e.g., 224x224 for many pre-trained models)
    img = image.resize((224, 224))
    
    # Convert the image to an array of pixel values
    img_array = np.asarray(img)
    
    # Normalize pixel values to be in the range [0, 1]
    img_array = img_array / 255.0
    
    # Expand dimensions to create a batch (add a dimension for batch size)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_class(image):
    processed_image = preprocess_image(image)

    # Make predictions using the loaded Keras model
    predictions = model.predict(processed_image)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class label
    predicted_class = class_names[predicted_class_index]

    return predicted_class

def run():
    st.title('Image Prediction from URL')
    st.write('It can predict image like building, forest, glacier, mountain, sea, and street')
    image_url = st.text_input("Enter Image URL:")

    if image_url:
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                # Open the image from the URL
                image = Image.open(response.raw)

                # Display the image
                st.image(image, caption='Image from URL', use_column_width=True)

                if st.button('Predict'):
                    # Get the predicted class label
                    predicted_class = predict_class(image)

                    # Display the predicted class
                    st.write(f'Predicted class: {predicted_class}')

            else:
                st.write("Unable to fetch image from URL.")
        except Exception as e:
            st.write(f"Error occurred: {e}")

# Run the Streamlit app
if __name__ == '__main__':
    run()
