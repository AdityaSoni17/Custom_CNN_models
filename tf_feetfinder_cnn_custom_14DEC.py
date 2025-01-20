import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models ,regularizers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pyheif
#from tensorflow.keras.callbacks import Progbar
from tensorflow.keras.utils import Progbar
from tensorflow.keras import backend as K
#from imblearn.over_sampling import SMOTE
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.debugging.set_log_device_placement(False)
#tf.get_logger().setLevel('ERROR')
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_SIZE = (512, 512)  # Resize all images to 512x512

# Enable dynamic memory growth for TensorFlow to allocate memory as needed
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Using experimental API for setting memory growth in TensorFlow 2.x
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info("Memory growth enabled for GPU.")
    except RuntimeError as e:
        logger.error(f"Error enabling memory growth: {e}")
tf.debugging.set_log_device_placement(True)
tf.keras.backend.clear_session()

# Function to check if a file is an image by extension
def is_image_file(filename):
    """
    Check if the given filename is an image file by its extension.
    """
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.heic']
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

# Convert HEIC images to JPEG format
def heic_to_jpeg(heic_path):
    """
    Convert a HEIC file to a JPEG file.
    """
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride, heif_file.size[1]
    )
    jpeg_path = heic_path.replace('.heic', '.jpg')
    image.save(jpeg_path, 'JPEG')
    return jpeg_path


# Load image paths and labels for training
logger.info("Loading training data.")
train_image_paths = []
train_labels = []
train_dir = 'feetfinder_dataset/train'

# Traverse through subdirectories (Feet, Non_feet)
for class_name in ['feet', 'not_feet']:
    class_dir = os.path.join(train_dir, class_name)
    for filename in os.listdir(class_dir):
        if is_image_file(filename):
            file_path = os.path.join(class_dir, filename)
            if file_path.lower().endswith('.heic'):
                file_path = heic_to_jpeg(file_path)  # Convert HEIC to JPEG
            train_image_paths.append(file_path)
            # Assign label based on the subdirectory name
            label = 0 if class_name == 'feet' else 1
            train_labels.append(label)

# Load image paths and labels for testing
logger.info("Loading testing data.")
test_image_paths = []
test_labels = []
test_dir = 'feetfinder_dataset/test'

# Traverse through subdirectories (Feet, Non_feet)
for class_name in ['feet', 'not_feet']:
    class_dir = os.path.join(test_dir, class_name)
    for filename in os.listdir(class_dir):
        if is_image_file(filename):
            file_path = os.path.join(class_dir, filename)
            if file_path.lower().endswith('.heic'):
                file_path = heic_to_jpeg(file_path)  # Convert HEIC to JPEG
            test_image_paths.append(file_path)
            # Assign label based on the subdirectory name
            label = 0 if class_name == 'feet' else 1
            test_labels.append(label)

# Convert to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Image data generator for training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.2],
    channel_shift_range=5.0
)

# Image data generator for testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255#,
    # rotation_range=30,
    # width_shift_range=0.3,
    # height_shift_range=0.3,
    # shear_range=0.2,
    # zoom_range=0.3,
    # horizontal_flip=True,
    # fill_mode='nearest',
    # brightness_range=[0.8, 1.3],
    # channel_shift_range=10.0
    )


# Load the training and testing datasets
train_generator = train_datagen.flow_from_directory(
    'feetfinder_dataset/train/',
    target_size=(512, 512),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
# Create a dataset from the ImageDataGenerator
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,  # The generator function
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),  # Image shape
        tf.TensorSpec(shape=(None,), dtype=tf.float32)               # Label shape
    )
)
test_generator = test_datagen.flow_from_directory(
    'feetfinder_dataset/test/',
    target_size=(512,512),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)
# Create a dataset from the ImageDataGenerator
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,  # The generator function
    output_signature=(
        tf.TensorSpec(shape=(None, 512,512, 3), dtype=tf.float32),  # Image shape
        tf.TensorSpec(shape=(None,), dtype=tf.float32)               # Label shape
    )
)


# Get the first batch from the training and testing datasets
for i, (image_batch, label_batch) in enumerate(train_dataset):
    if i == 0:  # Take the first batch
        print(f"First batch of images shape: {image_batch.shape}")
        print(f"First batch of labels shape: {label_batch.shape}")
        break

# If you want to check for the test dataset too
for i, (image_batch, label_batch) in enumerate(test_dataset):
    if i == 0:  # Take the first batch
        print(f"First batch of test images shape: {image_batch.shape}")
        print(f"First batch of test labels shape: {label_batch.shape}")
        break
#
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dense(512,activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])



def build_custom_model(input_shape=(512, 512, 3)):
    model = models.Sequential()

    # 1st Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 4th Convolutional Block (optional, only if needed)
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 5th Convolutional Block (optional, only if needed)
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output from the convolutional layers
    model.add(layers.Flatten())

    # Fully connected layers
    # model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.Dropout(0.3))

    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification (feet or not)

    return model

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_arch = build_custom_model()
model_arch.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy', AUC(), Precision(), Recall()]
)
logger.info("Training the model...")
history = model_arch.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=test_generator.samples // test_generator.batch_size,
    validation_data=test_dataset,
    epochs=50,
    callbacks=[lr_scheduler,early_stopping],
)

# Save the trained model
model_arch.save('tf_model_custom_cnn_v2_14Dec.h5')
import tensorflow as tf
import tensorflow.keras.backend as K

# After training
K.clear_session()  # Clear the Keras session and free memory

# Optionally, you can force garbage collection
import gc
gc.collect()

#tf.keras.backend.clear_session()

# Ensure the predictions are flattened
y_pred_prob = model_arch.predict(test_dataset)  # Predictions from the model
y_pred = (y_pred_prob > 0.5) # Convert to binary (0/1) and flatten
print(f"prediction : {y_pred}")
# Ensure y_true is the correct shape (flattened array of integers)
y_true = test_generator.classes#.flatten()

# Check the shape of y_true and y_pred
print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
logger.info(f"Confusion Matrix:\n{cm}")

# Compute ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)  # Use probabilities here
roc_auc = auc(fpr, tpr)
logger.info(f"AUC: {roc_auc:.4f}")

# Classification Report
logger.info("Classification Report:")
logger.info(classification_report(y_true, y_pred))  # Use binary predictions here


# Function to save ROC curve plot
def save_roc_curve(fpr, tpr, auc, filename):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()

save_roc_curve(fpr, tpr, roc_auc, 'roc_curve_v1.png')

# Evaluate the model on the test set (additional metrics)
logger.info("Evaluating the model on the test set...")
test_loss, test_acc, test_auc, test_precision, test_recall = model_arch.evaluate(test_dataset)

logger.info(f"Test Accuracy: {test_acc:.4f}")
logger.info(f"Test AUC: {test_auc:.4f}")
logger.info(f"Test Precision: {test_precision:.4f}")
logger.info(f"Test Recall: {test_recall:.4f}")

