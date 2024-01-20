import os
import cv2
import pickle
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import *
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from IPython.display import HTML
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tqdm import tqdm


# # Constants
# IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
# SEQUENCE_LENGTH = 16
# CLASSES_LIST = ["NonViolence", "Violence"]

# # Functions

# def play_video(filepath):
#     """Play video inline in Jupyter Notebook"""
#     html = ''
#     video = open(filepath, 'rb').read()
#     src = 'data:video/mp4;base64,' + b64encode(video).decode()
#     html += '<video width=640 muted controls autoplay loop><source src="%s" type="video/mp4"></video>' % src
#     return HTML(html)

def extract_frames(video_path, SEQUENCE_LENGTH=16, IMAGE_HEIGHT=64, IMAGE_WIDTH=64):
    """
        Extract frames from video file and return a list of frames
    """
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the number of frames to skip between each frame
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    # Extract frames from video
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        
        if not success:
            break
        
        # Resize frame to IMAGE_HEIGHTxIMAGE_WIDTH size
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def create_dataset(CLASSES_LIST, DATASET_DIR, SEQUENCE_LENGTH=16):
    """Create dataset from videos and return features and labels"""
    features = []
    labels = []
    video_files_paths = []
    
    # Create features and labels from videos
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = extract_frames(video_file_path)
            
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels, video_files_paths

def preprocess_labels(labels):
    # one-hot encode labels
    return to_categorical(labels)

def split_dataset(features, labels):
    return train_test_split(features, labels, test_size=0.1, shuffle=True, random_state=42)

def build_model(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH):
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    for layer in mobilenet.layers[:40]:
        layer.trainable = False
    
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(units=32, backward_layer=LSTM(units=32, go_backwards=True))))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    return model

def save_model_and_history(model, history):
    model.save('model.h5')
    with open('model_history.pkl', 'wb') as file:
        pickle.dump(history, file)


def train_model(model, x_train, y_train):
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=5, min_lr=0.00005, verbose=1)
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy", tensorflow.keras.metrics.AUC()])
    
    history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=8, shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback, reduce_lr])

    save_model_and_history(model, history)
    return history



def predict_video(video_file_path, model, CLASSES_LIST):
    frames_list = extract_frames(video_file_path)
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    print(f'Predicted: {predicted_class_name}\probability: {predicted_labels_probabilities[predicted_label]}')


def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def get_acuracy_score(y_true, y_pred):
    return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

def plot_class_labels(y_true, y_pred, CLASSES_LIST):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(CLASSES_LIST, np.sum(y_true, axis=0))
    axes[0].set_title("True Labels")
    axes[1].bar(CLASSES_LIST, np.sum(y_pred, axis=0))
    axes[1].set_title("Predicted Labels")
    plt.show()

def plot_model_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Accuracy_Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss_curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.show()

def plot_model_architecture(model):
    plot_model(model, show_shapes=True, show_layer_names=True)

def plot_class_distribution(labels, CLASSES_LIST):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(CLASSES_LIST, np.bincount(labels))
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class Name")
    ax.set_ylabel("Count")
    plt.show()

def plot_video_frames_with_labels(video_file_path, model, CLASSES_LIST):
    frames_list = extract_frames(video_file_path)
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_labels = np.argmax(predicted_labels_probabilities, axis=1)
    fig, axes = plt.subplots(2, 8, figsize=(15, 5))
    for i in range(2):
        for j in range(8):
            axes[i, j].imshow(frames_list[i * 8 + j])
            axes[i, j].axis('off')
            axes[i, j].set_title(CLASSES_LIST[predicted_labels[i * 8 + j]])
    plt.show()





# # Main Workflow
# features, labels, video_files_paths = create_dataset()
# one_hot_encoded_labels = preprocess_labels(labels)
# x_train, x_test, y_train, y_test = split_dataset(features, one_hot_encoded_labels)
# model = build_model()
# history = train_model(model, x_train, y_train)

# # Example of predicting a video
# predict_video(input_video_file_path)
# play_video(input_video_file_path)

# preds = model.predict(x_test)
# plot_confusion_matrix(y_test, preds)
# print(f'Accuracy Score: {get_acuracy_score(y_test, preds)}')
# plot_class_labels(y_test, preds)
# plot_model_history(history)
# plot_model_architecture(model)
# plot_class_distribution(labels)
# plot_video_frames_with_labels(input_video_file_path, model)

