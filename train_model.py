import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


def load_and_preprocess_data():
    """
    Carrega o dataset MNIST e realiza a normalização e reshape
    adequados para a entrada de uma CNN.
    """
    print("INFO: Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalização dos tensores (0-255 para 0.0-1.0)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Adicionando o canal de cor (grayscale)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)


def build_edge_cnn_model():
    """
    Constrói uma arquitetura CNN otimizada para dispositivos Edge.
    Utiliza poucos filtros para minimizar o uso de memoria e CPU,
    atendendo as restricoes do ambiente de CI/CD.
    """
    print("INFO: Construindo arquitetura da CNN...")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_edge_cnn_model()

    print("\nINFO: Iniciando o treinamento...")
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    print("\nINFO: Avaliando o modelo...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"INFO: Acurácia final no conjunto de teste: {test_acc * 100:.2f}%")

    model_path = 'model.h5'
    model.save(model_path)
    print(f"INFO: Modelo salvo com sucesso em: {model_path}")


if __name__ == "__main__":
    main()
