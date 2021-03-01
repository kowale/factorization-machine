import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from time import time
from matplotlib import pyplot as plt
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

def batcher(X, y=None, batch_size=-1, randomize=False):
    n_samples = X.shape[0]
    if batch_size == -1: batch_size = n_samples
    if randomize:
        for i in range(n_samples // batch_size):
            indices = np.random.choice(np.arange(n_samples), batch_size)
            batch_x, batch_y = X[indices], y[indices]
            yield batch_x, batch_y
    else:
        for i in range(0, n_samples, batch_size):
            upper = min(n_samples, i+batch_size)
            batch_x = X[i:upper]
            if y is None:
                batch_y = None
            else:
                batch_y = y[i:upper]
            yield batch_x, batch_y

def feed(X):
    X = X.tocoo()
    i = torch.from_numpy(np.hstack(((
        X.row[:, np.newaxis]),
        X.col[:, np.newaxis])).astype(np.int64)).long()
    v = torch.from_numpy(X.data.astype(np.float32)).float()
    s = torch.Size(np.array(X.shape).astype(np.int64))
    return torch.sparse.FloatTensor(i.t(), v, s)

X = load_npz("data/X.npz")
y = np.load("data/y.npy")

n, k = X.shape[1], 10
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

b = Variable(torch.zeros(1), requires_grad=True)
w = Variable(torch.zeros(n, 1), requires_grad=True)
V = Variable(0.01*torch.randn(n, k), requires_grad=True)

th = lambda A: torch.from_numpy(A).float()
linear = lambda X: b + X.mm(w).reshape(-1)
pairwise = lambda X: 0.5*(X.mm(V).pow(2) - X.pow(2).mm(V.pow(2))).sum(1)
predict = lambda X: linear(X) + pairwise(X)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([b, w, V], lr=0.001)

test_losses = []
train_losses = []

for epoch in range(250):
    for batch_x, batch_y in batcher(X_train, y_train, batch_size=200):
        y_pred = predict(feed(batch_x))
        l1 = 0.001 * torch.norm(torch.cat([t.view(-1) for t in [w, V]]), 1)
        l2 = 0.001 * torch.norm(torch.cat([t.view(-1) for t in [w, V]]), 2)
        loss = criterion(y_pred, th(batch_y)) + l1 + l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_losses.append(criterion(predict(feed(X_test)), th(y_test)))
    train_losses.append(loss.item())
    if epoch % 10 == 0:
        print('Epoch', epoch, 'Train loss', loss.item(), \
                'Test loss', criterion(predict(feed(X_test)), th(y_test)).item())

epoch_count = range(1, len(train_losses) + 1)
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(epoch_count, train_losses, 'r--')
plt.plot(epoch_count, test_losses, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f'figures/loss_{int(time())}.png', bbox_inches='tight')
plt.show()
