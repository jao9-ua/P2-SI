from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time

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
                accuracy_per_digit = []
                time_per_digit = []
                for digit in range(10):
                    # Balancear el conjunto de entrenamiento para el dígito actual
                    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, digit)
                    y_train_balanced = np.where(y_train_balanced == digit, 1, -1)
                    y_test_digit = np.where(y_test == digit, 1, -1)
                    
                    start_time = time.time()
                    
                    clf = AdaboostBinario(T=T, A=A)
                    clf.fit(X_train_balanced, y_train_balanced)
                    y_pred = clf.predict(X_test)
                    
                    accuracy = accuracy_score(y_test_digit, y_pred)
                    elapsed_time = time.time() - start_time
                    
                    accuracy_per_digit.append(accuracy)
                    time_per_digit.append(elapsed_time)
                
                accuracies.append(np.mean(accuracy_per_digit))
                times.append(np.mean(time_per_digit))
            
            results['T'].append(T)
            results['A'].append(A)
            results['accuracy'].append(np.mean(accuracies))
            results['time'].append(np.mean(times))
    
    return results

def plot_results(results, fixed_param, param_name):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for param in np.unique(results[fixed_param]):
        subset = [i for i in range(len(results[fixed_param])) if results[fixed_param][i] == param]
        plt.plot([results[param_name][i] for i in subset], [results['accuracy'][i] for i in subset], label=f'{fixed_param}={param}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs {param_name} for different {fixed_param} values')
    plt.legend()

    plt.subplot(1, 2, 2)
    for param in np.unique(results[fixed_param]):
        subset = [i for i in range(len(results[fixed_param])) if results[fixed_param][i] == param]
        plt.plot([results[param_name][i] for i in subset], [results['time'][i] for i in subset], label=f'{fixed_param}={param}')
    plt.xlabel(param_name)
    plt.ylabel('Time (s)')
    plt.title(f'Time vs {param_name} for different {fixed_param} values')
    plt.legend()

    plt.tight_layout()
    plt.show()

class AdaboostMulticlase:
    def __init__(self, T=5, A=20, n_components=50):
        self.T = T
        self.A = A
        self.classifiers = []
        self.classes = []
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y):
        X_reduced = self.pca.fit_transform(X)
        self.classes = np.unique(y)
        self.classifiers = []
        
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            clf = AdaboostBinario(T=self.T, A=self.A)
            clf.fit(X_reduced, y_binary)
            self.classifiers.append(clf)

    def predict(self, X):
        X_reduced = self.pca.transform(X)
        clf_preds = np.array([clf.predict(X_reduced) for clf in self.classifiers])
        return self.classes[np.argmax(clf_preds, axis=0)]

def train_and_evaluate_multiclase(X_train, y_train, X_test, y_test):
    clf = AdaboostMulticlase(T=10, A=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Clasificador Multiclase:")
    print("Matriz de Confusión:")
    print(conf_mat)
    print(f"Precisión: {accuracy * 100:.2f}%")
    
    plt.matshow(conf_mat)
    plt.title('Matriz de Confusión para el clasificador multiclase')
    plt.colorbar()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def experiment_multiclase(T_values, A_values, X_train, y_train, X_test, y_test):
    results = {'T': [], 'A': [], 'accuracy': [], 'time': []}
    
    for T in T_values:
        for A in A_values:
            accuracies = []
            times = []
            for _ in range(5):  # Ejecutar cinco veces para promediar
                start_time = time.time()
                
                clf = AdaboostMulticlase(T=T, A=A)
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

def train_and_evaluate_multiclase_pca(X_train, y_train, X_test, y_test, n_components):
    clf = AdaboostMulticlase(T=100, A=50, n_components=n_components)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Clasificador Multiclase con PCA (n_components={n_components}):")
    print("Matriz de Confusión:")
    print(conf_mat)
    print(f"Precisión: {accuracy * 100:.2f}%")
    
    plt.matshow(conf_mat)
    plt.title(f'Matriz de Confusión para el clasificador multiclase (n_components={n_components})')
    plt.colorbar()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def experiment_multiclase_pca(T_values, A_values, X_train, y_train, X_test, y_test, n_components):
    results = {'T': [], 'A': [], 'accuracy': [], 'time': []}
    
    for T in T_values:
        for A in A_values:
            accuracies = []
            times = []
            for _ in range(5):  # Ejecutar cinco veces para promediar
                start_time = time.time()
                
                clf = AdaboostMulticlase(T=T, A=A, n_components=n_components)
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

    #1A
    #ClasificadorAdaBoost(X_train, y_train, X_test, y_test)

    #1B
    #T_values = [10, 30]
    #A_values = [10, 90]
    #results_comb = experiment(T_values, A_values, X_train, y_train, X_test, y_test)
    #plot_results(results_comb, 'T', 'A')

    #1C
    #train_and_evaluate_multiclase(X_train, y_train, X_test, y_test)

    # Experimentar con diferentes valores de T y A
    #T_values = [10, 20, 30, 90]
    #A_values = [10, 20, 30, 30]
    #results = experiment_multiclase(T_values, A_values, X_train, y_train, X_test, y_test)
    #plot_results(results, 'T', 'A')

    # Entrenar y evaluar el clasificador multiclase con reducción de dimensionalidad
    n_components = 50  # Número de componentes principales
    train_and_evaluate_multiclase_pca(X_train, y_train, X_test, y_test, n_components)

    # Experimentar con diferentes valores de T y A con reducción de dimensionalidad
    T_values = [30, 60, 70]
    A_values = [30, 40, 50]
    results = experiment_multiclase_pca(T_values, A_values, X_train, y_train, X_test, y_test, n_components)
    plot_results(results, 'T', 'A')

if __name__ == "__main__":
    main()
