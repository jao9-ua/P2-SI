import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

class DecisionStump:
    def __init__(self, n_features):
        # Seleccionar al azar una característica, un umbral y una polaridad.
        self.feature_index = np.random.randint(0, n_features)
        self.threshold = np.random.random()
        self.polarity = 1 if np.random.random() > 0.5 else -1

    def predict(self, X):
        # Si la característica es mayor que el umbral y la polaridad es 1,
        # o si es menor que el umbral y la polaridad es -1, devolver 1 (pertenece a la clase).
        # Si no, devolver -1 (no pertenece a la clase).
        feature_values = X[:, self.feature_index]
        if self.polarity == 1:
            return np.where(feature_values >= self.threshold, 1, -1)
        else:
            return np.where(feature_values < self.threshold, 1, -1)

class AdaboostBinario:
    def __init__(self, T=5, A=20):
        # Dar valores a los parámetros del clasificador e iniciar la lista de clasificadores débiles vacía
        self.T = T
        self.A = A
        self.stumps = []
        self.alphas = []

    def fit(self, X, y, verbose=False):
        # Obtener el número de observaciones y de características por observación de X
        n_samples, n_features = X.shape
        
        # Iniciar pesos de las observaciones a 1/n_observaciones
        weights = np.ones(n_samples) / n_samples
        
        # Bucle de entrenamiento Adaboost: desde 1 hasta T repetir
        for t in range(self.T):
            min_error = float('inf')
            best_stump = None

            # Bucle de búsqueda de un buen clasificador débil: desde 1 hasta A repetir
            for _ in range(self.A):
                # Crear un nuevo clasificador débil aleatorio
                stump = DecisionStump(n_features)
                
                # Calcular predicciones de ese clasificador para todas las observaciones
                predictions = stump.predict(X)
                
                # Calcular el error: comparar predicciones con los valores deseados y acumular los pesos de las observaciones mal clasificadas
                error = np.sum(weights[(predictions != y)])
                
                # Actualizar mejor clasificador hasta el momento: el que tenga menor error
                if error < min_error:
                    min_error = error
                    best_stump = stump

            # Calcular el valor de alfa y las predicciones del mejor clasificador débil
            best_predictions = best_stump.predict(X)
            alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))

            # Actualizar pesos de las observaciones en función de las predicciones, los valores deseados y alfa
            weights *= np.exp(-alpha * y * best_predictions)
            
            # Normalizar a 1 los pesos
            weights /= np.sum(weights)
            
            # Guardar el clasificador en la lista de clasificadores de Adaboost
            self.stumps.append(best_stump)
            self.alphas.append(alpha)

    def predict(self, X):
        # Calcular las predicciones de cada clasificador débil para cada input multiplicadas por su alfa
        stump_preds = np.array([alpha * stump.predict(X) for alpha, stump in zip(self.alphas, self.stumps)])
        
        # Sumar para cada input todas las predicciones ponderadas y decidir la clase en función del signo
        return np.sign(np.sum(stump_preds, axis=0))

def balance_dataset(X, y, target_class):
    # Encuentra las muestras de la clase objetivo
    X_target = X[y == target_class]
    y_target = y[y == target_class]
    
    # Encuentra las muestras de las demás clases
    X_other = X[y != target_class]
    y_other = y[y != target_class]
    
    # Balancear las otras clases
    unique_classes = np.unique(y_other)
    n_samples_target = len(y_target)
    n_samples_per_class = n_samples_target // len(unique_classes)
    
    X_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        X_cls = X_other[y_other == cls]
        X_cls = X_cls[:n_samples_per_class]
        y_cls = y_other[y_other == cls]
        y_cls = y_cls[:n_samples_per_class]
        X_balanced.append(X_cls)
        y_balanced.append(y_cls)
    
    # Combina las muestras balanceadas de las otras clases con la clase objetivo
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)
    
    X_balanced = np.vstack((X_balanced, X_target))
    y_balanced = np.hstack((y_balanced, y_target))
    
    return X_balanced, y_balanced

def ClasificadorAdaBoost(X_train, y_train, X_test, y_test):
    accuracies = []

    # Entrenar un clasificador AdaboostBinario para cada dígito
    for digit in range(10):
        # Balancear el conjunto de entrenamiento
        X_balanced, y_balanced = balance_dataset(X_train, y_train, digit)
        y_balanced = np.where(y_balanced == digit, 1, -1)
        
        clf = AdaboostBinario(T=10, A=20)
        clf.fit(X_balanced, y_balanced)
        y_test_digit = np.where(y_test == digit, 1, -1)
        y_pred = clf.predict(X_test)
        
        conf_mat = confusion_matrix(y_test_digit, y_pred)
        accuracy = accuracy_score(y_test_digit, y_pred)
        accuracies.append(accuracy)

        print(f"Clasificador para el dígito {digit}:")
        print("Matriz de Confusión:")
        print(conf_mat)
        print(f"Precisión: {accuracy * 100:.2f}%")
        
        plt.matshow(conf_mat)
        plt.title(f'Matriz de Confusión para el dígito {digit}')
        plt.colorbar()
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.show()

    # Mostrar la precisión promedio para todas las clases
    print(f"Precisión promedio para todas las clases: {np.mean(accuracies) * 100:.2f}%")

def experiment(T_values, A_values, X_train, y_train, X_test, y_test):
    results = {'T': [], 'A': [], 'accuracy': [], 'time': []}
    
    for T in T_values:
        for A in A_values:
            accuracies = []
            times = []
            for _ in range(5):  # Ejecutar cinco veces para promediar
                start_time = time.time()
                
                clf = AdaboostBinario(T=T, A=A)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                elapsed_time = time.time() - start_time
                
                accuracies.append(accuracy)
                times.append(elapsed_time)
            
            results['T'].append(T)
            results['A'].append(A)
            results['accuracy'].append(np.mean(accuracies))
            results['time'].append(np.mean(times))
    
    return results

def plot_results(results):
    T_values = np.unique(results['T'])
    A_values = np.unique(results['A'])
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for A in A_values:
        subset = [i for i in range(len(results['A'])) if results['A'][i] == A]
        plt.plot(T_values, [results['accuracy'][i] for i in subset], label=f'A={A}')
    plt.xlabel('T')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs T for different A values')
    plt.legend()

    plt.subplot(1, 2, 2)
    for A in A_values:
        subset = [i for i in range(len(results['A'])) if results['A'][i] == A]
        plt.plot(T_values, [results['time'][i] for i in subset], label=f'A={A}')
    plt.xlabel('T')
    plt.ylabel('Time (s)')
    plt.title('Time vs T for different A values')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Función principal para entrenar y mostrar resultados
def main():
    # Cargar datos MNIST desde sklearn
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"].astype(np.int8)

    # Asegurarse de que X y y sean arrays de NumPy
    X = np.array(X)
    y = np.array(y)

    # Normalizar las características
    X = X / 255.0

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    #ClasificadorAdaBoost(X_train, y_train, X_test, y_test)

    T_values = [10, 20, 30]
    A_values = [10, 20, 30]
    
    y_train_digit = np.where(y_train == 0, 1, -1)
    y_test_digit = np.where(y_test == 0, 1, -1)
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, 0)
    y_train_balanced = np.where(y_train_balanced == 0, 1, -1)

    results = experiment(T_values, A_values, X_train_balanced, y_train_balanced, X_test, y_test_digit)
    plot_results(results)

if __name__ == "__main__":
    main()
