import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a arquitetura do MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Dados de treinamento para a porta XOR
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Instancia a MLP
input_size = 2
hidden_size = 2
output_size = 1
mlp = MLP(input_size, hidden_size, output_size)

# Define o otimizador e a função de perda
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.01)

# Treina a MLP
epochs = 1000
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = mlp(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
end_time = time.time()

print(f"Tempo total de treinamento: {end_time - start_time} segundos")

# Valida a MLP
with torch.no_grad():
    for i in range(len(x)):
        predicted = mlp(x[i]).round().item()
        print(f"Entrada: {x[i].numpy()}, Saída esperada: {y[i].item()}, Saída prevista: {predicted}")
