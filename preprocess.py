import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import csc_matrix
from scipy.sparse import save_npz

print('Loading reviews')
df = pd.read_csv('data/books_subset.csv')

print('Sampling subset')
cm = df.sample(8000)

print('One-hot encoding users')
user_encoder = LabelEncoder()
integer_encoded_users = user_encoder.fit_transform(cm.reviewerID)
onehot_encoded_users = OneHotEncoder(sparse=True).fit_transform(
        integer_encoded_users.reshape(-1, 1))

print('One-hot encoding items')
product_encoder = LabelEncoder()
integer_encoded_products = product_encoder.fit_transform(cm.asin)
onehot_encoded_products = OneHotEncoder(sparse=True).fit_transform(
        integer_encoded_products.reshape(-1, 1))

f = lambda z: (lambda x, y: 1 if y == 0 else (1+x/y**1.8)**y)(*eval(z))

y = np.array(cm.overall, dtype=np.float32)
X = hstack([
    onehot_encoded_users,
    onehot_encoded_products,
    csc_matrix(np.vstack(cm.helpful.apply(f)))
    ])

print('Saving processed data')
np.save("data/y.npy", y)
save_npz("data/X.npz", X)
