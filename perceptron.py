import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializa pesos com valores aleatórios pequenos
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _mean_squared_error(self, target, output):
        return np.mean((target - output) ** 2)

    def forward_pass(self, x):
        # Passagem da entrada pela camada escondida
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)

        # Passagem da camada escondida pela camada de saída
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)
        
        return self.final_output

    def backward_pass(self, x, y, output):
        # Calcula o erro da saída
        output_error = y - output
        output_delta = output_error * self._sigmoid_derivative(output)

        # Calcula o erro da camada escondida
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # Atualiza pesos e bias
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += x.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, x, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward_pass(x)
            self.backward_pass(x, y, output)
            if epoch % 1000 == 0:
                loss = self._mean_squared_error(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        output = self.forward_pass(x)
        return np.where(output >= 0.5, 1, 0)

# Dados de treinamento para a porta XOR
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Instancia a MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

# Treina a MLP
mlp.train(x, y, epochs=10000)

# Valida a MLP
for i in range(len(x)):
    print(f"Entrada: {x[i]}, Saída esperada: {y[i]}, Saída prevista: {mlp.predict(x[i])}")
