from tensorflow.keras.datasets import mnist
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
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
        self.T = T
        self.A = A
        self.stumps = []
        self.alphas = []

    def fit(self, X, y, X_val, y_val):
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples
        best_val_accuracy = 0
        best_t = 0

        for t in range(self.T):
            min_error = float('inf')
            best_stump = None

            for _ in range(self.A):
                stump = DecisionStump(n_features)
                predictions = stump.predict(X)
                error = np.sum(weights[(predictions != y)])
                
                if error < min_error:
                    min_error = error
                    best_stump = stump

            best_predictions = best_stump.predict(X)
            alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))
            weights *= np.exp(-alpha * y * best_predictions)
            weights /= np.sum(weights)
            
            self.stumps.append(best_stump)
            self.alphas.append(alpha)

            # Validación
            val_predictions = self.predict(X_val, stop_at_t=t+1)
            val_accuracy = accuracy_score(y_val, np.sign(val_predictions))
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_t = t
            elif t - best_t > 10:  # Si no mejora en 10 iteraciones, parar
                print(f"Parada temprana en la iteración {t + 1}")
                break

    def predict(self, X, stop_at_t=None):
        if stop_at_t is None:
            stop_at_t = len(self.stumps)
        stump_preds = np.array([alpha * stump.predict(X) for alpha, stump in zip(self.alphas[:stop_at_t], self.stumps[:stop_at_t])])
        return np.sum(stump_preds, axis=0)

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
            X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X_reduced, y_binary, test_size=0.2, random_state=42)
            clf = AdaboostBinario(T=self.T, A=self.A)
            clf.fit(X_train_cls, y_train_cls, X_val_cls, y_val_cls)
            self.classifiers.append(clf)

    def predict(self, X):
        X_reduced = self.pca.transform(X)
        clf_preds = np.array([clf.predict(X_reduced) for clf in self.classifiers])
        return self.classes[np.argmax(clf_preds, axis=0)]

def experiment_multiclase_early_stop(T_values, A_values, X_train, y_train, X_test, y_test, n_components):
    results = {'T': [], 'A': [], 'accuracy': [], 'time': []}
    
    for T in T_values:
        for A in A_values:
            if T * A > 3600:
                continue
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

def train_and_evaluate_multiclase_early_stop(X_train, y_train, X_test, y_test, n_components):
    clf = AdaboostMulticlase(T=100, A=50, n_components=n_components)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Clasificador Multiclase con parada temprana (n_components={n_components}):")
    print("Matriz de Confusión:")
    print(conf_mat)
    print(f"Precisión: {accuracy * 100:.2f}%")
    
    plt.matshow(conf_mat)
    plt.title(f'Matriz de Confusión para el clasificador multiclase con parada temprana (n_components={n_components})')
    plt.colorbar()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

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

def train_and_evaluate_multiclase_early_stop(X_train, y_train, X_test, y_test, n_components):
    clf = AdaboostMulticlase(T=100, A=50, n_components=n_components)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Clasificador Multiclase con parada temprana (n_components={n_components}):")
    print("Matriz de Confusión:")
    print(conf_mat)
    print(f"Precisión: {accuracy * 100:.2f}%")
    
    plt.matshow(conf_mat)
    plt.title(f'Matriz de Confusión para el clasificador multiclase con parada temprana (n_components={n_components})')
    plt.colorbar()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

def experiment_multiclase_early_stop(T_values, A_values, X_train, y_train, X_test, y_test, n_components):
    results = {'T': [], 'A': [], 'accuracy': [], 'time': []}
    
    for T in T_values:
        for A in A_values:
            if T * A > 3600:
                continue
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

def train_and_evaluate_sklearn_adaboost(X_train, y_train, X_test, y_test, T=50, A=20):
    # Inicializar el clasificador débil como un árbol de decisión con profundidad 1 y max_features=A
    weak_clf = DecisionTreeClassifier(max_depth=1, max_features=A)
    
    # Inicializar el clasificador Adaboost con el clasificador débil
    clf = AdaBoostClassifier(estimator=weak_clf, n_estimators=T)
    
    # Entrenar el clasificador Adaboost
    clf.fit(X_train, y_train)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = clf.predict(X_test)
    
    # Calcular la matriz de confusión y la precisión
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Mostrar los resultados
    print(f"Clasificador Adaboost de sklearn (T={T}, A={A}):")
    print("Matriz de Confusión:")
    print(conf_mat)
    print(f"Precisión: {accuracy * 100:.2f}%")
    
    plt.matshow(conf_mat)
    plt.title(f'Matriz de Confusión para Adaboost de sklearn (T={T}, A={A})')
    plt.colorbar()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    return accuracy

def experiment_sklearn_adaboost(T_values, A_values, X_train, y_train, X_test, y_test):
    results = {'T': [], 'A': [], 'accuracy': [], 'time': []}
    
    for T in T_values:
        for A in A_values:
            start_time = time.time()
            
            accuracy = train_and_evaluate_sklearn_adaboost(X_train, y_train, X_test, y_test, T=T, A=A)
            
            elapsed_time = time.time() - start_time
            
            results['T'].append(T)
            results['A'].append(A)
            results['accuracy'].append(accuracy)
            results['time'].append(elapsed_time)
    
    return results

def plot_sklearn_results(results):
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

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel='linear', C=1.0):
    # Inicializar el clasificador SVM con los parámetros dados
    clf = SVC(kernel=kernel, C=C)

    # Entrenar el clasificador SVM
    start_time = time.time()
    clf.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    # Realizar predicciones en el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Calcular la matriz de confusión y la precisión
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Mostrar los resultados
    print(f"Clasificador SVM (kernel={kernel}, C={C}):")
    print("Matriz de Confusión:")
    print(conf_mat)
    print(f"Precisión: {accuracy * 100:.2f}%")
    print(f"Tiempo de entrenamiento: {elapsed_time:.2f} segundos")

    plt.matshow(conf_mat)
    plt.title(f'Matriz de Confusión para SVM (kernel={kernel}, C={C})')
    plt.colorbar()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

    return accuracy, elapsed_time

def experiment_svm(kernel_values, C_values, X_train, y_train, X_test, y_test):
    results = {'kernel': [], 'C': [], 'accuracy': [], 'time': []}
    
    for kernel in kernel_values:
        for C in C_values:
            accuracy, elapsed_time = train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel=kernel, C=C)
            
            results['kernel'].append(kernel)
            results['C'].append(C)
            results['accuracy'].append(accuracy)
            results['time'].append(elapsed_time)
    
    return results

def plot_svm_results(results):
    kernel_values = np.unique(results['kernel'])
    C_values = np.unique(results['C'])
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for kernel in kernel_values:
        subset = [i for i in range(len(results['kernel'])) if results['kernel'][i] == kernel]
        plt.plot(C_values, [results['accuracy'][i] for i in subset], label=f'kernel={kernel}')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs C for different kernel values')
    plt.legend()

    plt.subplot(1, 2, 2)
    for kernel in kernel_values:
        subset = [i for i in range(len(results['kernel'])) if results['kernel'][i] == kernel]
        plt.plot(C_values, [results['time'][i] for i in subset], label=f'kernel={kernel}')
    plt.xlabel('C')
    plt.ylabel('Time (s)')
    plt.title('Time vs C for different kernel values')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Función principal para entrenar y mostrar resultados
def main():
    # Cargar datos MNIST desde keras
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Asegurarse de que X y y sean arrays de NumPy y normalizar las características
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

    #1A
    #ClasificadorAdaBoost(X_train, y_train, X_test, y_test)

    #1B
    '''T_values = [10, 30]
    A_values = [10, 90]
    results_comb = experiment(T_values, A_values, X_train, y_train, X_test, y_test)
    plot_results(results_comb, 'T', 'A')'''

    #1C
    '''train_and_evaluate_multiclase(X_train, y_train, X_test, y_test)

    # Experimentar con diferentes valores de T y A
    T_values = [10, 90]
    A_values = [10, 30]
    results = experiment_multiclase(T_values, A_values, X_train, y_train, X_test, y_test)
    plot_results(results, 'T', 'A')'''

    #1D
    '''# Entrenar y evaluar el clasificador multiclase con reducción de dimensionalidad
    n_components = 90  # Número de componentes principales
    train_and_evaluate_multiclase_pca(X_train, y_train, X_test, y_test, n_components)

    # Experimentar con diferentes valores de T y A con reducción de dimensionalidad
    T_values = [10, 30]
    A_values = [10, 90]
    results = experiment_multiclase_pca(T_values, A_values, X_train, y_train, X_test, y_test, n_components)
    plot_results(results, 'T', 'A')'''

    #1E
    '''# Entrenar y evaluar el clasificador multiclase con reducción de dimensionalidad y parada temprana
    n_components = 50  # Número de componentes principales
    train_and_evaluate_multiclase_early_stop(X_train, y_train, X_test, y_test, n_components)

    # Experimentar con diferentes valores de T y A con reducción de dimensionalidad y parada temprana
    T_values = [10, 90]
    A_values = [10, 30]
    results = experiment_multiclase_early_stop(T_values, A_values, X_train, y_train, X_test, y_test, n_components)
    plot_results(results, 'T', 'A')'''

    #2A
    '''T_values = [10, 90]
    A_values = [10, 30]
    results_sklearn = experiment_sklearn_adaboost(T_values, A_values, X_train, y_train, X_test, y_test)
    plot_sklearn_results(results_sklearn)'''

    #2B
    kernel_values = ['linear', 'rbf']
    C_values = [0.1, 1, 10]
    results_svm = experiment_svm(kernel_values, C_values, X_train, y_train, X_test, y_test)
    plot_svm_results(results_svm)

if __name__ == "__main__":
    main()
