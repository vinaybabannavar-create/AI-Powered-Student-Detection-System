# train_dummy_model.py
# Builds and saves a small sklearn model to models/trained_model.pkl for demo purposes
import os, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

os.makedirs('models', exist_ok=True)
MODEL_PATH = os.path.join('models', 'trained_model.pkl')

# Our fusion expects 10 (video) + 13 (audio) + 2 (text) = 25 dims
D = 25
N = 500
X = np.random.randn(N, D)
# make labels correlated slightly with audio feature (index ~ 10..22)
y = (X[:, 10] + X[:, 11] > 0).astype(int)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(clf, f)
print("Dummy model saved to:", MODEL_PATH)
