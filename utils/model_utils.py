import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from keras._tf_keras.keras.applications import VGG16, MobileNet, ResNet50, EfficientNetB0
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


import matplotlib.pyplot as plt
from PIL import Image
import io
import streamlit as st
import time

# Constants
IMG_SIZE = 224
CHANNELS = 3
MODEL_DIR = "models/"
MODEL_FILE = "trained_model.keras"
DATA_DIR = "data/"

class TrainingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        # Update progress
        progress = int((epoch + 1) / self.params['epochs'] * 100)
        
        # Update stats in session state
        st.session_state['training_stats'] = {
            'progress': progress,
            'accuracy': float(logs.get('accuracy', 0.0)),
            'loss': float(logs.get('loss', 0.0)),
            'val_accuracy': float(logs.get('val_accuracy', 0.0)),
            'val_loss': float(logs.get('val_loss', 0.0))
        }
        
        # Update display
        stats_container = st.session_state.get('stats_container')
        if stats_container:
            stats_container.markdown(f'''
            <div class="training-stats">
                <div class="stat-card">
                    <div class="stat-value">{progress}%</div>
                    <div class="stat-label">Progress</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{logs.get("accuracy", 0.0):.2%}</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{logs.get("val_accuracy", 0.0):.2%}</div>
                    <div class="stat-label">Val Acc</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{logs.get("loss", 0.0):.4f}</div>
                    <div class="stat-label">Loss</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Update progress bar
        progress_bar = st.session_state.get('progress_bar')
        if progress_bar:
            progress_bar.progress(progress / 100)

def create_model(num_classes, model_name='VGG16', dropout_rate=0.3):
    """Create a model using selected architecture with efficient training setup"""
    # Dictionary of available models
    model_architectures = {
        'VGG16': VGG16,
        'ResNet50': ResNet50,
        'MobileNet': MobileNet,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0
    }
    
    # Get the selected model architecture
    if model_name not in model_architectures:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_architectures.keys())}")
    
    base_model_class = model_architectures[model_name]
    
    # Load pre-trained model
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)
    )
    
    # Freeze all layers in the base model
    base_model.trainable = False
    
    # Create the model with efficient architecture
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        
        # Rescaling layer
        layers.Rescaling(1./255),
        
        # Base model
        base_model,
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Efficient classifier head
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_data(data_dir, batch_size=32, data_augmentation=True):
    """Prepare data with efficient augmentation"""
    if data_augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
            validation_split=0.2
        )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train_model(model_name='VGG16', epochs=25, batch_size=32, learning_rate=0.001, 
                dropout_rate=0.3, data_augmentation=True, early_stopping=True):
    """Train the model with optimized single-phase training"""
    try:
        # Check for GPU availability
        device = '/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'
        print(f"Training on: {device}")
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Prepare data
        train_generator, validation_generator = prepare_data(DATA_DIR, batch_size, data_augmentation)
        num_classes = len(train_generator.class_indices)
        
        with tf.device(device):
            # Create and compile model
            model = create_model(num_classes, model_name, dropout_rate)
            
            # Use a higher learning rate since we're only training the top layers
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Setup callbacks
            callbacks = [TrainingCallback()]
            
            if early_stopping:
                callbacks.append(EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    min_delta=0.01
                ))
            
            callbacks.append(ModelCheckpoint(
                os.path.join(MODEL_DIR, MODEL_FILE),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ))
            
            # Train the model
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save class indices and history
            class_indices = train_generator.class_indices
            class_indices = {v: k for k, v in class_indices.items()}
            np.save(os.path.join(MODEL_DIR, 'class_indices.npy'), class_indices)
            np.save(os.path.join(MODEL_DIR, 'training_history.npy'), history.history)
            
            # Return final metrics
            return {
                'accuracy': float(history.history['accuracy'][-1]),
                'loss': float(history.history['loss'][-1]),
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'val_loss': float(history.history['val_loss'][-1])
            }
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        st.error(f"Training failed: {str(e)}")
        return None

def predict_image(image_file):
    """Predict class for a given image with improved preprocessing"""
    try:
        # Load model and class indices
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        if not os.path.exists(model_path):
            print("Model file not found")
            return None, 0.0
        
        model = models.load_model(model_path)
        class_indices_path = os.path.join(MODEL_DIR, 'class_indices.npy')
        if not os.path.exists(class_indices_path):
            print("Class indices file not found")
            return None, 0.0
            
        class_indices = np.load(class_indices_path, allow_pickle=True).item()
        
        # Prepare image
        try:
            # First try to read the image directly
            img = Image.open(image_file)
            # Ensure the image is in RGB mode
            img = img.convert('RGB')
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            # If direct reading fails, try reading from bytes
            if hasattr(image_file, 'read'):
                img = Image.open(io.BytesIO(image_file.read()))
                img = img.convert('RGB')
            else:
                print("Failed to open image file")
                return None, 0.0
        
        # Resize with antialiasing
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        img_array = np.array(img)
        img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction with test-time augmentation
        predictions = []
        
        # Original image prediction
        pred = model.predict(img_array, verbose=0)
        predictions.append(pred)
        
        # Horizontal flip
        flipped = tf.image.flip_left_right(img_array)
        pred = model.predict(flipped, verbose=0)
        predictions.append(pred)
        
        # Slight rotation
        rotated = tf.image.rot90(img_array, k=1)
        pred = model.predict(rotated, verbose=0)
        predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        class_idx = np.argmax(avg_pred[0])
        confidence = float(avg_pred[0][class_idx])
        
        # Return class name and confidence
        return class_indices[class_idx], confidence
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        st.error(f"Error processing image: {str(e)}")
        return None, 0.0

def plot_training():
    """Plot training history with improved visualization"""
    try:
        history_path = os.path.join(MODEL_DIR, 'training_history.npy')
        if not os.path.exists(history_path):
            print("Training history not found")
            return None
            
        history = np.load(history_path, allow_pickle=True).item()
        
        if history:
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(12, 8))
            
            # Plot accuracy
            plt.subplot(2, 2, 1)
            plt.plot(history['accuracy'], label='Training', linewidth=2)
            plt.plot(history['val_accuracy'], label='Validation', linewidth=2)
            plt.title('Model Accuracy', pad=20)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot loss
            plt.subplot(2, 2, 2)
            plt.plot(history['loss'], label='Training', linewidth=2)
            plt.plot(history['val_loss'], label='Validation', linewidth=2)
            plt.title('Model Loss', pad=20)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot additional metrics
            if 'auc' in history:
                plt.subplot(2, 2, 3)
                plt.plot(history['auc'], label='Training', linewidth=2)
                plt.plot(history['val_auc'], label='Validation', linewidth=2)
                plt.title('Model AUC', pad=20)
                plt.xlabel('Epoch')
                plt.ylabel('AUC')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            if 'precision' in history:
                plt.subplot(2, 2, 4)
                plt.plot(history['precision'], label='Training', linewidth=2)
                plt.plot(history['val_precision'], label='Validation', linewidth=2)
                plt.title('Model Precision', pad=20)
                plt.xlabel('Epoch')
                plt.ylabel('Precision')
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            return plt
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")
        return None

