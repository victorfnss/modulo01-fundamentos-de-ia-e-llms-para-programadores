# Exemplo 00 - Recomendação com TensorFlow.js

Este projeto é um exemplo prático inicial do uso de **Inteligência Artificial** com **TensorFlow.js**. 
O objetivo deste código é criar e treinar um modelo simples de rede neural capaz de categorizar usuários em perfis de consumo (`premium`, `medium`, `basic`) com base em características como:
- Idade
- Cor favorita
- Localização

## Estrutura

- `index.js`: Contém o script que define as camadas da rede neural, os dados de treino normalizados, compila o modelo usando o otimizador *Adam* e o utiliza para prever o perfil de um novo usuário.
- `package.json` / `package-lock.json`: Definem as dependências do projeto (focado no pacote `@tensorflow/tfjs-node`).

## Como Executar

1. Certifique-se de ter o [Node.js](https://nodejs.org/) instalado.
2. Baixe as dependências do projeto:
   ```bash
   npm install
   ```
3. Execute o script principal:
   ```bash
   node index.js
   ```

O script irá treinar a rede e em seguida fará a predição da categoria do usuário demonstrado no código, imprimindo as porcentagens de chance para cada categoria.
