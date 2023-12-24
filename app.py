from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import os
import requests
from werkzeug.utils import secure_filename
from tempfile import TemporaryDirectory
from keras.models import Model

app = Flask(__name__)

# Load your trained feature extraction model
inception_model = InceptionV3(weights='imagenet', include_top=False)
feature_model = Model(inputs=inception_model.input, outputs=inception_model.get_layer('mixed10').output)

# Google Drive URL of your classifier model
model_url = 'https://drive.google.com/uc?export=download&id=1ISEPcs61rq4RuqSnPiX0I59b3FY-NJmz'


# Function to download the model from Google Drive
def download_model(url, model_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(model_path, 'wb') as file:
            file.write(response.content)
    else:
        raise Exception('Failed to download model')

# Path to save the downloaded model
classifier_model_path = './multi_class_model_new.keras'

# Download and load your trained classifier model
download_model(model_url, classifier_model_path)
classifier_model = load_model(classifier_model_path)

@app.route('/', methods=['GET'])
def index():
    # Render the upload HTML file
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the file to a temporary directory
        with TemporaryDirectory() as temp_dir:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            # Preprocess the image file
            img = load_img(temp_path, target_size=(299, 299))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            # Extract features
            features = feature_model.predict(img)
            
            # Flatten the features
            flattened_features = features.flatten().reshape(1, -1)
            
            # Make prediction using the classifier model
            prediction = classifier_model.predict(flattened_features)
            predicted_class = np.argmax(prediction, axis=1)

            # Map the predicted class to its label
            class_labels = {1: "Chanterelle", 0: "Non-Chanterelle", 2: "False-Chanterelle"}
            result = class_labels.get(predicted_class[0], "Unknown")
            
            return result

if __name__ == '__main__':
    app.run(debug=True)
