import os
import shutil
import numpy as np
import kagglehub
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import label_binarize
import random

# =====================================================
# 1. BAIXAR DATASET
# =====================================================
path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
print("Dataset baixado em:", path)

train_src = os.path.join(path, "train")
test_src = os.path.join(path, "test")
classes = ["Goat", "Giraffe"]

# =====================================================
# 2. CRIAR ESTRUTURA FINAL
# =====================================================
base_dir = "dataset_animais"

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# =====================================================
# 3. FUNÇÃO DE SPLIT (SEM DATA LEAKAGE)
# =====================================================
def copiar_dados_corrigido():
    print("\nCopiando dados corretamente...")

    for cls in classes:
        origem = os.path.join(train_src, cls)
        imgs = [
            f for f in os.listdir(origem)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(imgs)

        split_idx = int(0.85 * len(imgs))
        train_imgs = imgs[:split_idx]
        val_imgs = imgs[split_idx:]

        for f in train_imgs:
            shutil.copy(os.path.join(origem, f), os.path.join(base_dir, "train", cls))

        for f in val_imgs:
            shutil.copy(os.path.join(origem, f), os.path.join(base_dir, "val", cls))

    # Copia o TESTE original inteiro
    for cls in classes:
        origem = os.path.join(test_src, cls)
        dest = os.path.join(base_dir, "test", cls)

        for f in os.listdir(origem):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy(os.path.join(origem, f), dest)

    print("Cópia concluída sem misturar os conjuntos!\n")

copiar_dados_corrigido()

# =====================================================
# 4. CARREGAR DATASETS
# =====================================================
img_size = (224, 224)
batch = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_dir, "train"),
    image_size=img_size,
    batch_size=batch
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_dir, "val"),
    image_size=img_size,
    batch_size=batch
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(base_dir, "test"),
    image_size=img_size,
    batch_size=batch,
    shuffle=False
)

# 🟢 CORREÇÃO QUE GARANTE AMBAS AS CLASSES NO PRIMEIRO BATCH
test_ds = test_ds.unbatch().shuffle(5000, seed=42).batch(batch)


# Pré-processamento
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# =====================================================
# 5. DATA AUGMENTATION
# =====================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# =====================================================
# 6. MODELO (MobileNetV2)
# =====================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation="softmax")  # 2 classes fixo
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# =====================================================
# 7. TREINO FASE 1
# =====================================================
callback = tf.keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[callback]
)

# =====================================================
# 8. FINE TUNING
# =====================================================
base_model.trainable = True

for layer in base_model.layers[:20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[callback]
)

# =====================================================
# 9. AVALIAÇÃO FINAL
# =====================================================
loss, acc = model.evaluate(test_ds)
print(f"\nAcurácia no TESTE REAL: {acc:.4f}")

# =====================================================
# 10. MATRIZ DE CONFUSÃO
# =====================================================
y_true = []
y_pred = []
y_scores = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_scores.extend(preds)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# =====================================================
# 11. RELATÓRIO DE CLASSIFICAÇÃO
# =====================================================
report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
df_report = pd.DataFrame(report).transpose()

print("\nRelatório de Classificação:")
print(df_report)

plt.figure(figsize=(8, 4))
sns.heatmap(df_report.iloc[:-1, :-1].astype(float), annot=True, cmap="Purples")
plt.title("Métricas: Precision, Recall, F1-score")
plt.show()

# =====================================================
# 12. CURVA ROC / AUC (FUNCIONA SEMPRE)
# =====================================================
y_true_bin = label_binarize(y_true, classes=[0, 1])
y_scores = np.array(y_scores)

# Garante 2 colunas SEMPRE
if y_scores.shape[1] == 1:
    print("\n[AVISO] Modelo retornou apenas 1 coluna. Convertendo...")
    y_scores = np.hstack([1 - y_scores, y_scores])

plt.figure(figsize=(7, 6))

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("Curva ROC / AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# =====================================================
# 13. VISUALIZAÇÃO DE PREVISÕES
# =====================================================
plt.figure(figsize=(10, 10))
for i in range(9):
    img_batch, label_batch = next(iter(test_ds))
    idx = random.randint(0, len(img_batch) - 1)
    img = img_batch[idx].numpy().astype("uint8")

    pred = model.predict(img_batch[idx:idx+1])
    pred_label = classes[np.argmax(pred)]

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"Real: {classes[label_batch[idx]]}\nPred: {pred_label}")
    plt.axis("off")

plt.show()

print("\nExecução finalizada com sucesso!")
