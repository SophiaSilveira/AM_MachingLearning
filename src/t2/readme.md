# 🐐🦒 Classificação de Imagens — Goat vs Giraffe

Este projeto treina uma rede neural utilizando **Transfer Learning (MobileNetV2)** para classificar imagens de **Goat** e **Giraffe**.

Abaixo estão as instruções necessárias **apenas para executar o projeto**.


## 📦 1. Instalar Dependências

Antes de executar o projeto, instale os pacotes necessários:

```bash
pip install -r requirements.txt


## ▶️ 2. Executar o Script Principal

```bash
python3 main.py


### 📝 Observações
- O dataset será baixado automaticamente através do kagglehub.
- As pastas train/, val/ e test/ serão criadas automaticamente dentro de dataset_animais/.
- Ao final da execução, o script exibirá a accuracy, matriz de confusão e relatório de classificação.
- Não é necessário configurar nada manualmente — tudo é feito automaticamente pelo script.