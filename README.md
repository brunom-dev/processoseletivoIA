# Relatório Técnico: Otimização de Modelo para Edge AI

👤 **Identificação:** Bruno da Silva Macedo

---

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo implementado no arquivo `train_model.py` é uma Rede Neural Convolucional (CNN) sequencial leve, projetada especificamente para eficiência computacional em dispositivos de borda. A arquitetura é composta por:
- **Camada de Entrada:** Recebe imagens em escala de cinza de 28x28 pixels.
- **Camadas de Extração de Features:** Duas camadas convolucionais (`Conv2D` com 16 e 32 filtros, respectivamente) utilizando a função de ativação ReLU, cada uma seguida por uma camada de subamostragem (`MaxPooling2D` 2x2) para redução de dimensionalidade.
- **Camadas de Classificação:** Uma etapa de achatamento (`Flatten`) que alimenta uma camada densa compacta de 64 neurônios, finalizando com uma camada de saída de 10 neurônios (função Softmax) para a classificação dos dígitos de 0 a 9.

### 2️⃣ Bibliotecas Utilizadas

- **Python** (v3.10 / 3.11)
- **TensorFlow / Keras** (v2.x): Utilizada para o carregamento do dataset MNIST, construção, treinamento e conversão do modelo matemático.
- **OS (Built-in)**: Utilizada para manipulação e leitura do tamanho final dos arquivos gerados.

### 3️⃣ Técnica de Otimização do Modelo

No arquivo `optimize_model.py`, foi aplicada a técnica de **Dynamic Range Quantization** (Quantização Dinâmica) nativa do conversor do TensorFlow Lite (`tf.lite.Optimize.DEFAULT`). 

Essa técnica reduz a precisão dos pesos matemáticos da rede neural de ponto flutuante de 32 bits (`float32`) para inteiros de 8 bits (`int8`), convertendo-os em tempo real durante a inferência. O objetivo alcançado foi diminuir drasticamente o peso do arquivo e o consumo de memória RAM, mantendo o modelo compatível com microcontroladores e dispositivos IoT sem perda significativa de acurácia.

### 4️⃣ Resultados Obtidos

- **Acurácia do Treinamento:** O modelo alcançou uma acurácia de 98.87% no conjunto de teste.
- **Eficiência de Borda (Redução de Tamanho):**
  - Tamanho original (`.h5`): **703.94 KB**
  - Tamanho otimizado (`.tflite`): **63.54 KB**
- **Impacto:** Obteve-se uma redução de aproximadamente 91% no tamanho de armazenamento do modelo, tornando-o altamente viável para embarque em hardware com recursos escassos.

### 5️⃣ Comentários Adicionais

A principal decisão técnica foi limitar o número de filtros (16 e 32) nas camadas convolucionais e manter a camada densa intermediária em apenas 64 neurônios. Evitar arquiteturas profundas garantiu que o modelo fosse treinado de forma extremamente rápida em CPU, respeitando as restrições de tempo de execução do pipeline de CI/CD (GitHub Actions).  