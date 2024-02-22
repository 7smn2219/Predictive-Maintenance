# prompt: test for all data and make table with filename and predicted label
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import librosa
from sklearn.preprocessing import LabelEncoder
from hdbcli import dbapi
import pickle


# Extract features
def extract_features(audio, sr=22050, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Average over time
    return mfccs


# Loctions and path
MODEL_PATH = '/app/model/model.pkl'
ok = "ok"
anomaly1 = "anomaly1"
anomaly2 = "anomaly2"

# LebelEncoder
le = LabelEncoder()
labels = le.fit_transform([ok,anomaly1, anomaly2])

# Load the saved model
model = pickle.load(open(MODEL_PATH, 'rb'))

print("Uploading to Datasphere ... ")

i , j = 0, 0
# Initialize an empty list to store the results
results = []
for folder in [ok,anomaly1, anomaly2]:
    for filename in os.listdir(os.path.join(os.getcwd(),"/app/data/Dataset/demo",folder)):
        print(filename)
        
        audio, sr = librosa.load(os.path.join("/app/data/Dataset/demo/", folder,filename))
        print("librosa loaded")

        features = extract_features(audio, sr=sr)

        # Make a prediction using the loaded model
        prediction = model.predict(features.reshape(1, features.shape[0], 1))

        # Get the predicted label
        predicted_label = le.inverse_transform(np.argmax(prediction, axis=1))

        # Check if the prediction is correct
        i+=1
        if(predicted_label[0] == folder):
            j+=1
        else:
            print(f"Predicted: {predicted_label[0]}, Actual: {folder}")
        # Add the filename and predicted label to the results list
        lable = predicted_label[0]
        file_c = filename[0:-4]
        results.append({'filename': file_c, 'predicted_label': lable})


# Print the accuracy
print(f"Accuracy: {j/i}")

# Convert the results list to a pandas DataFrame
results = pd.DataFrame(results)
# results.to_csv("results.csv")


print("Uploading to Datasphere ... ")
# UPLODING DATA TO DATASPHERE
conn = dbapi
conn = dbapi.connect(
    address="e7dcf416-50a3-4748-a430-913210d916ca.hna0.prod-us10.hanacloud.ondemand.com",
    port=443,
    user="CRAVEDEV#PM_USER",
    password="3!C.>6?GKF=.#`!V!@cinOB!i*m=_$i!"
)
cursor = conn.cursor()
# Replace 'your_table' with the actual table name where your data is stored
cursor.execute("Set schema CRAVEDEV#PM_USER")
cursor.execute("Drop table PM_PREDICTION")
cursor.execute("""
    CREATE TABLE PM_Prediction (
        filename varchar(255),
        predicted_label varchar(255)
    )
""")

# Prepare the data for insertion
data_to_insert = [tuple(row) for row in results[['filename','predicted_label']].itertuples(index=False)]

# Define the SQL insert statement
sql_insert = "INSERT INTO PM_PREDICTION(filename, predicted_label) VALUES (?, ?)"

# Use executemany for faster insertion
cursor.executemany(sql_insert, data_to_insert)

# Commit the changes
conn.commit()
