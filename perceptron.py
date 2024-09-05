import random

class Perceptron:
    def _init_(self,num_inputs):
        self.weights = [random.uniform(-1,1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1,1)

    def predict(self,inputs):
        activation = self.bias
        for i in range(len(inputs)):
            activation += inputs[i] * self.weights[i]
        return 1 if activation >= 0 else 0 #funcion de activacion
      
    def train(self, inputs, target):
        output = self.predict(inputs)
        error = target - output
        self.bias += error
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i]

    def get_weights(self):
        return self.weights
    
    def save_weights(self, filename):
        with open(filename, "w") as f:
            f.write(f"{self.bias}\n")
            for w in self.weights:
                f.write(f"{w}\n")

    def load_weights(self, filename):
        with open(filename, "r") as f:
            self.bias = float(f.readline())
            self.weights = [float(line) for line in f.readlines()]

# and
# x y -> z
# 0 0 -> 0
# 0 1 -> 1
# 1 0 -> 1
# 1 1 -> 0

#ejemplo de uso
perceptron = Perceptron(2)

# si existe el archivo, cargar los pasos
try:
    perceptron.load_weights("weights.txt")
except:
    print("No se pudo cargar los pesos")
    print("Entrenando...")
    #Entrenar
    for _ in range (1000):
        inputs = [0, 0]
        perceptron.train(inputs, 0)
        inputs = [0, 1]
        perceptron.train(inputs, 0)
        inputs = [1, 0]
        perceptron.train(inputs, 0)
        inputs = [1, 1]
        perceptron.train(inputs, 1)

    #guardar los pesos
    perceptron.save_weights("weights.txt")

print("Este es el modelo entrenado: ")
print(perceptron.get_weights())

print("Predicciones")
print([0, 0], "-" , perceptron.predict([0, 0]))
print([0, 1], "-" , perceptron.predict([0, 1]))
print([1, 0], "-" , perceptron.predict([1, 0]))
print([1, 1], "-" , perceptron.predict([1, 1]))

