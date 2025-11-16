# Trabalho - Interpretabilidade de Modelos de Aprendizado de Máquina

## Etapas

- [v] **1. Escolha do Dataset**
  - [V] Escolher dataset público (UCI, Kaggle, Google Dataset Search, etc.)
  - [V] Dataset deve ter pelo menos 5 features
  - [v] Variável predita deve ser categórica (classificação) - Variavel "Rate your academic stress index" (1 a 5) 
  - [v] Deve conter variáveis numéricas e categóricas
  - [v] Ter pelo menos 100 instâncias
  - [v] Não pode ser dataset já usado em aula (Iris, Titanic, Adult, Breast Cancer, Wine Quality, etc.)

- [ ] **2. Treinamento dos Modelos**
  - [v] Pré-processamento dos dados  
    - [V] Tratamento de valores ausentes  -> 1 valor ausente campo Study Environment, removido pois representa menos de 5% das instâncias
    - [v] Normalização  -> Não há dados fora do padrão nas features
    - [v] Codificação de variáveis categóricas 
        - Variaveís categóricas - será usado valore inteiros iniciados em 1 até o fim das categorias, 1 sempre será considerado o pior caso ou o que vem primeiro(no caso da escolaridade);
         1. Your Academic Stage -> 1(high school), 2(undergraduate), 3(post-graduate)
         2. Study Environment ->  1(Noisy), 2(disrupted), 3(Peaceful)
         3. What coping strategy you use as a student? -> 1(Emotional breakdown (crying a lot)), 2(Social support (friends, family)), 3(Analyze the situation and handle it with intellect)
         4. Do you have any bad habits like smoking, drinking on a daily basis? -> 1(No), 2(prefer not to say), 3(Yes)
  - [v] Divisão do dataset em treino e teste (80/20 ou 70/30)
  - [v] Treinar modelos:  
    - [v] KNN  
    - [v] Naïve Bayes  
    - [v] Árvore de Decisão  
  - [ ] Avaliar desempenho dos modelos com métricas:  
    - [ ] Acurácia  - entre 0.7 e 0.95
    - [ ] Precisão  - ente 0.6 e 0.95
    - [ ] Recall  - ente 0.6 e 0.95
    - [ ] F1-score  - ente 0.6 e 0.90
  - [v] Justificar escolhas de pré-processamento  
  - [ ] Justificar escolhas de treinamento
  - [ ] Garantir performance suficiente para análise de interpretabilidade

- [ ] **3. Interpretabilidade dos Modelos**
  - [ ] Árvore de Decisão: analisar a árvore e features mais importantes
  - [ ] Naïve Bayes: analisar probabilidades condicionais e sua influência
  - [ ] KNN: discutir dificuldade de interpretação e usar SHAP/LIME
  - [ ] Usar ferramentas adicionais (Análise de Permutação, SHAP, LIME, etc.)

- [ ] **4. Comparação e Análise**
  - [ ] Comparar interpretabilidade dos três modelos
  - [ ] Verificar se os resultados fizeram sentido
  - [ ] Identificar se os modelos concordaram nas variáveis mais relevantes
  - [ ] Explicar ferramentas de interpretabilidade utilizadas
  - [ ] Discutir limitações de cada modelo

- [ ] **5. Apresentação**
  - [ ] Gravar vídeo (10–15 minutos) apresentando o trabalho
  - [ ] Incluir no vídeo:
    - [ ] Descrição do dataset e do problema
    - [ ] Metodologia de treinamento e avaliação
    - [ ] Análise de interpretabilidade para cada modelo
    - [ ] Discussão comparativa e importância da interpretabilidade
    - [ ] Conclusões e reflexões finais

---

## Critérios de Avaliação
- [V] Escolha dos Dados – **1 ponto**
- [V] Pré-Processamento e Treinamento – **1 ponto**
- [ ] Interpretabilidade – **2 pontos**
- [ ] Análise – **3 pontos**
- [ ] Apresentação – **3 pontos**

---

## Entregáveis
- [ ] Código (GitHub ou pasta compactada no Moodle)
- [ ] Vídeo da apresentação (YouTube não listado ou outra plataforma) + arquivo `.txt` com o link
- [ ] Justificativas/discussões no código (README, comentários ou Markdown)

---

## Bibliotecas Recomendadas
- [ ] Scikit-learn
- [ ] Pandas
- [ ] NumPy
- [ ] Matplotlib
- [ ] Seaborn  
- [ ] SHAP  
- [ ] LIME  
- [ ] ELI5  

---

## Prazo
- [ ] Entregar até **30/09/2025**
