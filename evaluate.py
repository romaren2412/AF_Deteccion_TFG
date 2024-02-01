import torch

def evaluate_accuracy(data_loader, model, device):
    """
        Evalúa la precisión del modelo en el conjunto de datos.

        Parámetros:
            - data_iterator: Iterador de datos para el conjunto de datos.
            - net: Modelo a evaluar.
            - device: Dispositivo de ejecución (por ejemplo, 'cuda' para GPU, 'cpu' para CPU).

        Retorna:
            - accuracy: Precisión del modelo en el conjunto de datos.
        """
    model.eval()  # Cambiar el modo del modelo a evaluación
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    return accuracy


def evaluate_backdoor(data_iterator, net, target, device):
    # Igual que en MLR pero sin redimensionar a (-1, 784)
    """
    Evalúa la precisión del ataque en el conjunto de datos de puerta trasera (backdoor).
    (Compara, entre los ejemplos que inicialmente no eran de la clase destino,
    cuántos fueron clasificados correctamente como de la clase destino después de aplicar la puerta trasera)

    Parámetros:
    - data_iterator: Iterador de datos para el conjunto de datos.
    - net: Modelo a evaluar.
    - target: Clase de destino para la puerta trasera.
    - device: Dispositivo de ejecución (por ejemplo, 'cuda' para GPU, 'cpu' para CPU).

    Retorna:
    - accuracy: Precisión del ataque
    """
    net.eval()  # Establece el modelo en modo de evaluación
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Desactiva el cálculo del gradiente durante la evaluación
        for i, (data, label) in enumerate(data_iterator):
            data, label = data.to(device), label.to(device)

            # Aplica la puerta trasera a todas las imágenes para el test
            data[:, :, 26, 26] = 1
            data[:, :, 26, 24] = 1
            data[:, :, 25, 25] = 1
            data[:, :, 24, 26] = 1

            # Inicializa la lista de índices
            remaining_idx = list(range(data.shape[0]))

            # Se evalúa la precisión del ataque en los ejemplos que inicialmente NO eran de la clase destino.
            # "Elimina" los ejemplos que SÍ son de la clase de destino
            # Establece las etiquetas de los ejemplos que NO son de la clase de destino a la clase de destino
            # (objetivo del ataque)
            for example_id in range(data.shape[0]):
                if label[example_id] == target:
                    remaining_idx.remove(example_id)
                else:
                    label[example_id] = target

            # Propagación hacia adelante
            output = net(data)

            # Obtén las predicciones
            _, predictions = torch.max(output, 1)
            predictions = predictions[remaining_idx]
            label = label[remaining_idx]

            # Actualiza el conteo de predicciones correctas
            correct_predictions += (predictions == label).sum().item()
            total_samples += len(remaining_idx)

    accuracy = correct_predictions / total_samples
    return accuracy


def evaluate_edge_backdoor(data, net, device):
    """
    Evalúa la precisión del modelo en el conjunto de datos de puerta trasera (edge).

    Parámetros:
    - data: Datos de entrada.
    - net: Modelo a evaluar.
    - device: Dispositivo de ejecución (por ejemplo, 'cuda' para GPU, 'cpu' para CPU).

    Retorna:
    - accuracy: Precisión del modelo en el conjunto de datos de puerta trasera (edge).
    """
    net.eval()  # Establece el modelo en modo de evaluación
    with torch.no_grad():  # Desactiva el cálculo del gradiente durante la evaluación
        data = data.to(device)

        # Etiquetas para la puerta trasera (edge)
        label = torch.ones(len(data), dtype=torch.long).to(device)

        # Propagación hacia adelante
        output = net(data)

        # Obtén las predicciones
        _, predictions = torch.max(output, 1)

        # Evalúa la precisión
        correct_predictions = (predictions == label).sum().item()
        total_samples = len(data)
        accuracy = correct_predictions / total_samples

    return accuracy
