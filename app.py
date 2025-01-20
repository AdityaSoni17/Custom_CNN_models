from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
import imageio.v3 as imageio  # Use imageio.v2 to avoid deprecation warnings
import os
import tempfile
import io
import boto3
from decimal import Decimal
from flask_cors import CORS
import random
import numpy as np
import subprocess
import logging
from collections import Counter
from datetime import datetime
from werkzeug.middleware.proxy_fix import ProxyFix
from nudity_check import NudityCheck


app = Flask(__name__)
# Apply ProxyFix to handle proxy headers (e.g., X-Forwarded-For, X-Forwarded-Proto)
#app.wsgi_app = ProxyFix(app.wsgi_app)


model = load_model('tf_model_custom_cnn_v2_14Dec.h5')  # Load your model


# Apply the CORS middleware globally
#CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]}})
CORS(app, origins="*")
#CORS(app)
# Allow all origins, methods, and headers for all routes
"""
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Allow these methods
    "allow_headers": ["Content-Type", "Authorization"]  # Allow these headers
}})

# Enable CORS for all routes
CORS(app, resources={r"/*": {
    "origins": "*"#,  # Allow all origins (replace with specific origin if needed)
    #"methods": ["POST", "OPTIONS"],  # Allow POST and OPTIONS (for preflight)
    #"allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],  # Allow required headers
    #"allow_credentials": True  # Allow credentials if you need them (optional)
}})


CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins (could be restricted to your front-end domain in production)
    "methods": ["POST", "OPTIONS"],  # Allow only POST and OPTIONS for preflight
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],  # Allow necessary headers
    "allow_credentials": False  # Ensure this is False if you're not using credentials
}})
"""

#CORS(app, resources={r"/predict": {"origins": "*"}})


# Initialize AWS clients
AWS_REGION = 'us-east-1'  # Change to your actual region
s3_client = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table('FeetFinderPrediction')

# S3 Bucket name
S3_BUCKET = 'feetfinder-inputs-and-model'  # Your S3 bucket name


# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic']

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define max size for image resizing (512x512)
MAX_IMAGE_SIZE = (512, 512)

"""
@app.before_request
def before_request():
    #if request.headers.get('X-Forwarded-Proto') == 'https':
        request.url_scheme = 'https'
        request.is_secure = True


@app.before_request
def before_request():
    # Debugging: print headers to see if X-Forwarded-Proto exists
    print("Request Headers: ", request.headers)

    # Check if X-Forwarded-Proto is present
    if request.headers.get('X-Forwarded-Proto') == 'https':
        print("Request is secure (HTTPS).")
        # This line is not required but can be useful for debugging
        request.is_secure = True  # This line is optional, avoid it if ProxyFix is working
    else:
        print("Request is not secure (HTTP).")

"""

@app.after_request
def log_cors(response):
    # Log headers to verify CORS headers are set
    print(f"Response Headers: {response.headers}")
    return response

@app.route('/health')
def health_check():
    return "OK", 200


@app.route('/another-endpoint', methods=['GET'])
def another_endpoint():
    return jsonify({'message': 'Another endpoint response'})


def prepare_image(image):
    """
    #Preprocess the image to match the input shape expected by the model.
    """
    # Resize image to the target size (512x512)
    image = image.resize(MAX_IMAGE_SIZE)  # Adjust size based on your model's input

    # Convert to numpy array
    img_array = np.array(image)

    logging.debug(f"Image shape after resize: {img_array.shape}")

    # Check if the image is grayscale (2D) and convert to RGB by repeating the channel
    if len(img_array.shape) == 2:  # Grayscale image (height, width)
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to 3 channels (RGB)
        logging.debug("Converted grayscale to RGB")

    # Check if image has 4 channels (RGBA), remove the alpha channel (keeping RGB)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Drop the alpha channel
        logging.debug("Removed alpha channel")

    # Normalize image data to [0, 1] range
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    logging.debug(f"Image shape after normalization: {img_array.shape}")

    # Expand the dimensions of the image to include batch size (e.g., (1, 512, 512, 3))
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


def convert_heic_to_jpg(input_path, output_path):
    """
    #Convert a HEIC file to JPG using FFmpeg.
    """
    try:
        subprocess.run(['ffmpeg', '-i', input_path, output_path], check=True)
        logging.debug(f"HEIC file successfully converted to {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during HEIC to JPG conversion: {e}")
        raise


def handle_heic(image_file):
    """
    #Handle HEIC image file and convert it to a Pillow image using imageio or FFmpeg fallback.
    """
    try:
        # Create a temporary file to store the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.heic') as temp_file:
            temp_file.write(image_file)
            temp_file_path = temp_file.name

        logging.debug(f"Temporary file created at {temp_file_path}")

        # Try reading the HEIC file using imageio (without 'format' argument)
        try:
            image_data = imageio.imread(temp_file_path)  # Removed 'format' argument
            pil_image = Image.fromarray(image_data)  # Convert image to PIL format
        except Exception as e:
            logging.error(f"Error reading HEIC file with imageio: {str(e)}")
            # Fallback: Convert HEIC to JPG using FFmpeg
            output_path = temp_file_path.replace('.heic', '.jpg')
            convert_heic_to_jpg(temp_file_path, output_path)
            pil_image = Image.open(output_path)  # Open the converted image as PIL
            os.remove(output_path)  # Clean up converted file

        # Remove the temporary file after processing
        os.remove(temp_file_path)

        return pil_image

    except Exception as e:
        logging.error(f"Error processing HEIC file: {str(e)}")
        raise Exception(f"Error processing HEIC file: {str(e)}")

def tile_image(image, tile_size=(512, 512), overlap=0.2):
    """
    Tile the image into smaller patches with overlap between tiles.
    """
    img_width, img_height = image.size
    tiles = []
    step_size = int(tile_size[0] * (1 - overlap))  # Overlap step size

    for i in range(0, img_width - tile_size[0] + 1, step_size):
        for j in range(0, img_height - tile_size[1] + 1, step_size):
            tile = image.crop((i, j, i + tile_size[0], j + tile_size[1]))
            tiles.append(tile)

    # Ensure we capture the rightmost and bottommost parts of the image
    if img_width % tile_size[0] != 0:
        for j in range(0, img_height - tile_size[1] + 1, step_size):
            tile = image.crop((img_width - tile_size[0], j, img_width, j + tile_size[1]))
            tiles.append(tile)

    if img_height % tile_size[1] != 0:
        for i in range(0, img_width - tile_size[0] + 1, step_size):
            tile = image.crop((i, img_height - tile_size[1], i + tile_size[0], img_height))
            tiles.append(tile)

    return tiles

def majority_voting(predictions):
    """
    Use majority voting to determine the final prediction.
    """
    # Convert predictions to binary ('YES' or 'NO')
    binary_predictions = ['YES' if p <= 0.6 else 'NO' for p in predictions]

    # Find the most common prediction
    prediction_count = Counter(binary_predictions)
    final_prediction = prediction_count.most_common(1)[0][0]

    return final_prediction

def resize_with_padding(image, target_size=(512, 512)):
    img_width, img_height = image.size
    target_width, target_height = target_size

    # Calculate new dimensions while maintaining aspect ratio
    aspect_ratio = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * aspect_ratio)
    new_height = int(img_height * aspect_ratio)

    # Resize image
    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new image with padding
    new_image = Image.new("RGB", target_size, (0, 0, 0))  # Black padding
    new_image.paste(image_resized, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return new_image


# Function to generate random amount based on match confidence
def generate_fee(prediction_prob):
    """
    Generate fee based on prediction probability.

    :param prediction_prob: The predicted probability for the 'feet' class (0 to 0.6)
    :return: A random amount based on match category (Min, Moderate, Highest)
    """
    if prediction_prob < 0.2:
        # Minimum match (low confidence)
        fee_range = (0, 60)  # Random amount between 0 and 60
        fee_category = "Minimum Match"

    elif 0.2 <= prediction_prob < 0.4:
        # Moderate match (medium confidence)
        fee_range = (60, 120)  # Random amount between 60 and 120
        fee_category = "Moderate Match"

    elif 0.4 <= prediction_prob <= 0.6:
        # Highest match (high confidence)
        fee_range = (120, 200)  # Random amount between 120 and 200
        fee_category = "Highest Match"

    else:
        # Invalid case if probability is out of expected bounds (0 to 0.6)
        fee_range = (0,1)

    # Generate random fee within the determined range
    fee = round(random.uniform(fee_range[0], fee_range[1]), 2)

    return fee


# Example usage (predicting probability and then generating fee):
def predict_and_generate_fee(prediction_prob):
    """
    Predict the fee for the image based on the probability.

    :param prediction_prob: Predicted probability of the "feet" class (e.g. 0.45)
    :return: The fee and match category
    """
    fee = generate_fee(prediction_prob)

    if fee is not None:
        print(f"Prediction Probability: {prediction_prob}")
        # print(f"Category: {category}")
        print(f"Generated Fee: ${fee}")
        return fee
    else:
        print(f"Error: {category}")



def convert_to_native_float(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.float32):
        return float(obj)  # Convert np.float32 to float
    elif isinstance(obj, dict):
        return {k: convert_to_native_float(v) for k, v in obj.items()}  # Recursively handle dict
    elif isinstance(obj, list):
        return [convert_to_native_float(item) for item in obj]  # Recursively handle list
    else:
        return obj  # Return as is if it's already a serializable type



@app.route('/nudity_check',methods=['Post'])
def nudity_prediction():#image):
    try:
        logging.debug(f"Request Headers: {request.headers}")
        logging.debug(f"Request Body: {request.get_data(as_text=True)}")

        # Get image URL from the JSON request body
        data = request.get_json()

        image_url = data['image']

        nudity_check = NudityCheck()
        logging.debug(f"Image file : {image_url}")
        ratio = nudity_check.predict_image(image_path=image_url)
        result = convert_to_native_float(ratio)
        logging.debug(f"Prediction result: {ratio}")
        return jsonify({'nudity_rate':result['nudity_ratio'],'status':200}),200
    except Exception as e:
        logging.error(f"Nudity prediction is getting failed : {str(e)}")
        return jsonify({'error_message': str(e)
                }),500





@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Request received")
    logging.debug(f"Request headers: {request.headers}")
    logging.debug(f"Request content type: {request.content_type}")
    logging.debug(f"Request form: {request.form}")
    logging.debug(f"Request files: {request.files}")
    random.seed()
    random_number = round(random.uniform(4, 400), 2)

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_name = file.filename
    print(f"image name : {image_name}")

    # Check for supported file formats
    file_extension = image_name.lower().split('.')[-1]
    if file_extension not in [ext.lstrip('.') for ext in SUPPORTED_FORMATS]:
        return jsonify({
                           'error': f'Unsupported file type: {file_extension}. Supported types are: {", ".join(SUPPORTED_FORMATS)}'}), 400

    file_content = file.read()

    try:
        if file_extension == 'heic':
            image = handle_heic(file_content)
        else:
            image = Image.open(io.BytesIO(file_content))
    except Exception as e:
        return jsonify({'error': f'Error opening image file: {str(e)}'}), 500

        # Resize or tile the image if necessary
    if image.size[0] > 512 or image.size[1] > 512:
        image = resize_with_padding(image, target_size=(512, 512))
        img_array = prepare_image(image)
        prediction = model.predict(img_array)
        result = 'YES' if float(prediction[0][0]) <= 0.6 else 'NO'
        #tiles = tile_image(image, tile_size=(512, 512))
        #predictions = []
        #for tile in tiles:
        #    img_array = prepare_image(tile)
        #    prediction = model.predict(img_array)
        #    predictions.append(prediction[0][0])

        # Use majority voting to determine final result
        #result = majority_voting(predictions)

    else:
        img_array = prepare_image(image)
        prediction = model.predict(img_array)
        result = 'YES' if float(prediction[0][0]) <= 0.6 else 'NO'

    # Resize or tile the image if necessary
    #img_array = prepare_image(image)
    #prediction = model.predict(img_array)
    #result = 'YES' if float(prediction[0][0]) <= 0.5 else 'NO'

    # Upload to S3
    try:
        # Get current date and time
        current_datetime = datetime.now()

        # Format the date and time
        formatted_datetime = current_datetime.strftime(
            "%Y-%m-%d %H:%M:%S") + f".{current_datetime.microsecond // 1000:03d}"
        # Get the timestamp
        timestamp = int(current_datetime.timestamp() * 1000)
        s3_client.upload_fileobj(io.BytesIO(file_content), S3_BUCKET, str(timestamp) + image_name)
        file_url = f'https://{S3_BUCKET}.s3.amazonaws.com/{str(timestamp)}{image_name}'  # URL of the uploaded file
        print(f"File uploaded to S3: {file_url}, \n Prediction : {float(prediction[0][0])}")
    except Exception as e:
        return jsonify({'error': f'Error uploading to S3: {str(e)}'}), 500

    # Convert prediction to native float and then to Decimal
    prediction_value = float(prediction[0][0]) if isinstance(prediction[0][0], np.float64) else float(prediction[0][0])
    print(f"prediction_value : {prediction_value}")
    logging.info(f"Prediction value = {prediction_value} and real prediction is: {prediction}")
    print(f'rate : {predict_and_generate_fee(prediction_value)}')

    # Store prediction in DynamoDB
    response = table.put_item(
        Item={
            'image_key': str(timestamp)+image_name,
            'prediction': Decimal(str(prediction_value)),
            'predicted_class': result,
            'rate': str(predict_and_generate_fee(prediction_value)),
            's3_url': file_url
        }
    )
    print({'status': 200, 'image_name': str(timestamp)+image_name, 'prediction': result, 'rate':str(predict_and_generate_fee(prediction_value)),'s3_url': file_url})

    return jsonify({'status': 200, 'image_name': str(timestamp)+image_name, 'prediction': result, 'rate':str(predict_and_generate_fee(prediction_value)),
                    's3_url': file_url})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



