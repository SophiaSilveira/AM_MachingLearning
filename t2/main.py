import os
import shutil
import numpy as np
import kagglehub
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =====================================================
# 1. BAIXAR DATASET
# =====================================================
path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
print("Dataset baixado em:", path)

train_src = os.path.join(path, "train")
test_src = os.path.join(path, "test")

classes = ["Goat", "Giraffe"]

# =====================================================
# 2. CRIAR PASTAS DESTINO
# =====================================================
base_dir = "dataset_animais"
splits = ["train", "val", "test"]

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# =====================================================
# 3. FUNÇÃO PARA COPIAR SOMENTE IMAGENS
# =====================================================
def copiar_dados(origem_base):
    for cls in classes:
        origem = os.path.join(origem_base, cls)

        if not os.path.isdir(origem):
            print(f"Aviso: pasta não encontrada: {origem}")
            continue

        for fname in os.listdir(origem):
            src = os.path.join(origem, fname)

            # Ignorar subpastas (como "label")
            if os.path.isdir(src):
                continue

            # Filtrar apenas arquivos de imagem
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                continue

            # Split aleatório
            rand = np.random.rand()
            if rand < 0.7:
                dst = os.path.join(base_dir, "train", cls)
            elif rand < 0.85:
                dst = os.path.join(base_dir, "val", cls)
            else:
                dst = os.path.join(base_dir, "test", cls)

            shutil.copy(src, dst)

# Copiar datasets
copiar_dados(train_src)
copiar_dados(test_src)
print("Cópia concluída com sucesso!")

# =====================================================
# 4. CARREGAR DATASETS + PRÉ-PROCESSAMENTO
# =====================================================
img_size = (224, 224)
batch = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir + "/train",
    batch_size=batch,
    image_size=img_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir + "/val",
    batch_size=batch,
    image_size=img_size
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir + "/test",
    batch_size=batch,
    image_size=img_size
)

# Normalização do MobileNetV2
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess(x), y))

# Prefetch para acelerar
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# =====================================================
# 5. DATA AUGMENTATION
# =====================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.1),
])

# =====================================================
# 6. MOBILE NET V2 - TRANSFER LEARNING
# =====================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # primeiro congelamos

# Modelo inicial (feature extractor)
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# =====================================================
# 7. TREINAMENTO (FASE 1 - CONGELADO)
# =====================================================
callback = tf.keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[callback]
)

# =====================================================
# 8. FINE TUNING - DESCONGELAR CAMADAS SUPERIORES
# =====================================================
base_model.trainable = True

# congelar camadas iniciais (opcional, mas bom)
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[callback]
)

# =====================================================
# 9. AVALIAÇÃO FINAL
# =====================================================
loss, acc = model.evaluate(test_ds)
print(f"\nAccuracy no teste: {acc:.4f}")

# =====================================================
# 10. MATRIZ DE CONFUSÃO
# =====================================================
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nMatriz de confusão:")
print(confusion_matrix(y_true, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_true, y_pred, target_names=classes))
