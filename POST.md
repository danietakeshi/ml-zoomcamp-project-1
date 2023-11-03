Primeiro projeto do Machine Learning Zoomcamp : O Modelo de Previsão de Idade de Caranguejo

Para este desafio criei um modelo de machine learning com o objetivo de descobrir a idade de um caranguejo com base nos seus atributos físicos, uma vez que a partir de certo ponto os crustáceos deixam de ter crescimentos relevantes, portanto, nosso Modelo de Previsão de Idade de Caranguejo pode ser um divisor de águas para crustáceos do mundo real. Produtores comerciais de caranguejo, fiquem atentos! 🦐

Agora, sobre as metodologias de machine learning utilizadas no modelo de previsão de idade de caranguejo:

1. **Pré-processamento de Dados**: Antes de alimentar o modelo, os dados foram tratados e formatados. Isso incluiu a padronização de nomes de colunas, a eliminação de dados com altura zero e a aplicação de transformações nas unidades de medida.

2. **Engenharia de Recursos**: Foram criadas características adicionais para melhorar a precisão das previsões.

3. **Divisão dos Dados**: Os dados foram divididos em conjuntos de treinamento e teste para avaliar o desempenho do modelo.

4. **Modelo de Regressão com XGBoost**: O modelo de regressão XGBoost foi escolhido para prever a idade do caranguejo com base nas características selecionadas.

5. **Avaliação do Modelo**: O desempenho do modelo foi avaliado com base na métrica de erro médio quadrático (RMSE) entre as idades reais e as previstas.

6. **Deploy na Nuvem**: O modelo foi implantado na plataforma Google Cloud Run para disponibilizar previsões em um ambiente de produção acessível via API.

O projeto está disponível no Github (https://github.com/danietakeshi/ml-zoomcamp-project-1/tree/main).

Não posso dizer que o modelo de previsão é o melhor de todos, mas posso afirmar que caranguejo só é peixe na enchente da maré.

#machinelearning #mlzoomcamp 