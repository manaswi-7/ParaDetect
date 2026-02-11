import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

print("RUNNING FILE...")

# CHANGE THIS PATH if needed
data_dir = r"C:\Users\manas\Downloads\archive\cell_images"

categories = ["Parasitized", "Uninfected"]

img_size = 64

data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized = cv2.resize(img_array, (img_size, img_size))
            data.append(resized)
            labels.append(class_num)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Dataset Loaded Successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
