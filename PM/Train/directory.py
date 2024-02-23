import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf  # Example using TensorFlow
import pickle

ok = "ok"
anomaly1 = "anomaly1"
anomaly2 = "anomaly2"

print("Loading files...")

# Load wav files from folders, extract labels from folder names
data, labels = [], []
for folder in [ok,anomaly1, anomaly2]:
    print("in for loop with : ",folder)
    for filename in os.listdir(os.path.join(os.getcwd(),"/app/data/demo",folder)):
        print("in for loop with : ",filename)
