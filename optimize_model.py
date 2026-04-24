import os
import tensorflow as tf

def optimize_and_convert():
    """
    Carrega o modelo Keras (.h5) e aplica a técnica de 
    Dynamic Range Quantization para converter e otimizar
    o modelo para o formato TensorFlow Lite (.tflite),
    focando em inferência em dispositivos Edge.
    """
    model_path_h5 = 'model.h5'
    model_path_tflite = 'model.tflite'

    print(f"INFO: Carregando o modelo original ({model_path_h5})...")
    model = tf.keras.models.load_model(model_path_h5)

    print("INFO: Configurando o conversor do TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    print("INFO: Aplicando Dynamic Range Quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("INFO: Convertendo o modelo...")
    tflite_model = converter.convert()

    with open(model_path_tflite, 'wb') as f:
        f.write(tflite_model)

    # Verificacao de otimizacao
    size_h5_kb = os.path.getsize(model_path_h5) / 1024
    size_tflite_kb = os.path.getsize(model_path_tflite) / 1024
    
    print("\n--- Relatorio de Otimizacao ---")
    print(f"Tamanho original (.h5): {size_h5_kb:.2f} KB")
    print(f"Tamanho otimizado (.tflite): {size_tflite_kb:.2f} KB")
    print("Conversao concluida com sucesso!")

if __name__ == "__main__":
    optimize_and_convert()