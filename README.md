# README - Redes neuronales con numpy

---

## Descripción General

Este proyecto implementa una red neuronal desde cero utilizando **NumPy** y **Scikit-learn**, sin depender de frameworks avanzados como TensorFlow o Keras. La red neuronal se entrena para clasificar datos generados sintéticamente en dos clases. Este proyecto fue realizado con Python 3.12.

---

## Requisitos Previos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install numpy matplotlib scikit-learn
```

Opcionalmente, puedes usar un entorno virtual para aislar dependencias:

```bash
python -m venv env
source env/bin/activate  # En Linux/macOS
env\Scripts\activate  # En Windows
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
📂 neural_network_numpy/
├── 📂 src/                      
│   ├── neural_network_numpy.py   # Código fuente
├── main.py                       # Script principal para ejecutar el modelo
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Documentación del proyecto
```

---

## Implementación del Modelo

### Generación de Datos

Se generan datos sintéticos usando distribuciones gaussianas para clasificación binaria.

```python
from sklearn.datasets import make_gaussian_quantiles

X, Y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=2)
Y = Y[:, np.newaxis]  # Ajuste de dimensiones
```

### Preprocesamiento y Visualización

Los datos se visualizan en un gráfico de dispersión para comprobar su distribución.

```python
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title("Datos de Clasificación")
plt.show()
```

### Definición del Modelo

El modelo consta de:

- Tres capas ocultas con 6, 10 y 1 neurona(s) respectivamente.
- Funciones de activación **ReLU** y **Sigmoid**.

```python
def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x) / (np.exp(-x) + 1)**2
    return 1 / (1 + np.exp(-x))

def relu(x, derivate=False):
    if derivate:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)
```

### Inicialización de Parámetros

Los pesos y sesgos se inicializan aleatoriamente para cada capa de la red neuronal.

```python
def initialize_parameters_deep(layers_dims):
    parameters = {}
    for l in range(len(layers_dims) - 1):
        parameters[f'W{l+1}'] = np.random.rand(layers_dims[l], layers_dims[l+1]) * 2 - 1
        parameters[f'b{l+1}'] = np.random.rand(1, layers_dims[l+1]) * 2 - 1
    return parameters
```

### Entrenamiento del Modelo

La red neuronal se entrena mediante propagación hacia adelante y retropropagación.

```python
def train(x_data, learning_rate, params, training=True):
    params['A0'] = x_data
    
    # Forward propagation
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])
    
    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])
    
    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])
    
    output = params['A3']
    
    if training:
        # Backpropagation y actualización de parámetros
        params['dZ3'] = (output - Y) * sigmoid(output, True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])
        
        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])
        
        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])
        
        # Aplicar gradiente descendiente
        for i in range(1, 4):
            params[f'W{i}'] -= learning_rate * params[f'dW{i}']
            params[f'b{i}'] -= learning_rate * np.mean(params[f'dW{i}'], axis=0, keepdims=True)
    
    return output
```

Se ejecuta durante **50,000 iteraciones** para minimizar la función de pérdida.

```python
layers_dims = [2, 6, 10, 1]
params = initialize_parameters_deep(layers_dims)

for i in range(50000):
    output = train(X, 0.001, params)
    if i % 50 == 0:
        print(np.mean((output - Y)**2))
```

---

## Ejecución

Para entrenar y evaluar el modelo, ejecuta en la terminal:

```bash
python main.py
```

---

## Conclusión

Este proyecto muestra cómo se puede construir y entrenar una red neuronal simple utilizando exclusivamente NumPy y Scikit-learn, implementando manualmente la propagación hacia adelante y el algoritmo de retropropagación.
