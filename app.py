# from flask import Flask, request, render_template
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array, load_img
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
# import numpy as np
# import os
# from werkzeug.utils import secure_filename
# from tempfile import TemporaryDirectory
# from keras.models import Model

# app = Flask(__name__)

# # Load your trained feature extraction model
# inception_model = InceptionV3(weights='imagenet', include_top=False)
# feature_model = Model(inputs=inception_model.input, outputs=inception_model.get_layer('mixed10').output)  # Adjust to match the layer used during training

# # Load your trained classifier model
# classifier_model = load_model('./multi_class_model.keras')  # Load your trained classifier model here

# @app.route('/', methods=['GET'])
# def index():
#     # Render the upload HTML file
#     return render_template('upload.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return 'No file part'
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return 'No selected file'
    
#     if file:
#         # Save the file to a temporary directory
#         with TemporaryDirectory() as temp_dir:
#             filename = secure_filename(file.filename)
#             temp_path = os.path.join(temp_dir, filename)
#             file.save(temp_path)
            
#             # Preprocess the image file
#             img = load_img(temp_path, target_size=(299, 299))  # Adjust target_size if necessary
#             img = img_to_array(img)
#             img = np.expand_dims(img, axis=0)
#             img = preprocess_input(img)  # Use preprocess_input from InceptionV3
            
#             # Extract features
#             features = feature_model.predict(img)
            
#             # Flatten the features to match the input shape of the classifier model
#             flattened_features = features.flatten().reshape(1, -1)
            
#             # Make prediction using the classifier model
#             prediction = classifier_model.predict(flattened_features)
            
#             if prediction[0][0] > 0.5:
#                 result = "Chanterelle"
#             else:
#                 result = "Not Chanterelle"
            
#             return result

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
from tempfile import TemporaryDirectory
from keras.models import Model

app = Flask(__name__)

# Load your trained feature extraction model
inception_model = InceptionV3(weights='imagenet', include_top=False)
feature_model = Model(inputs=inception_model.input, outputs=inception_model.get_layer('mixed10').output)

# Load your trained classifier model
classifier_model = load_model('./multi_class_model_new.keras')

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
