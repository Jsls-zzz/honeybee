
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = np.array(range(11), dtype=np.float32).reshape(-1, 1)
y_train = np.array([2*i for i in range(11)], dtype=np.float32).reshape(-1, 1)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.Linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.Linear(x)
        return out

model = LinearRegressionModel(1, 1).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    epoch += 1

    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # loss.item() 从只有一个值的tensor中取出数据，多个值用tolist()
        print('epoch {}, loss {:.4f}'.format(epoch, loss.item()))

