# Projeto 03: WebAI - Inteligência Artificial no Navegador

Este projeto explora o uso prático da **WebAI** (Inteligência Artificial embarcada diretamente no navegador), focando na nova API experimental do Google Chrome (`window.LanguageModel` / API `ai.languageModel`). 

O objetivo principal deste projeto é demonstrar como criar aplicações que executam modelos de linguagem localmente (como o **Gemini Nano**), proporcionando inferências rápidas, maior privacidade para o usuário (os dados não saem do dispositivo) e funcionamento offline ou com latência zero.

A estrutura do projeto foi dividida em três subdiretórios que demonstram a evolução natural de uma aplicação utilizando essa API.

## Evolução das Versões

### 🔹 Versão 1 (`v1/`) - Prova de Conceito (PoC)
A primeira versão é uma implementação minimalista para validar a disponibilidade do modelo de linguagem no Chrome. 
- Utiliza a API `LanguageModel.availability()` para checar se o modelo suporta a língua portuguesa.
- Cria uma sessão básica de conversação (via `LanguageModel.create()`).
- Implementa a função de `promptStreaming`, permitindo exibir a resposta do modelo em tempo real no HTML, formatando de *Markdown* para HTML à medida que o texto é gerado.

### 🔹 Versão 2 (`v2/`) - Hyper-parâmetros e Interface de Usuário
A segunda versão introduz uma interface mais amigável e controles dinâmicos de comportamento do modelo.
- **Separação de responsabilidades:** O JavaScript é movido para um arquivo separado (`index.js`).
- **Controle de Temperatura (Temperature):** Adiciona um *slider* na interface que permite ajustar o quão criativa ou determinística será a resposta do modelo.
- **Controle de Top K:** Permite ao usuário definir o número de palavras mais prováveis que a IA deve considerar a cada passo da geração (controlando previsibilidade e coesão).
- **Controle de Geração:** Adição de interatividade para permitir cancelar/parar a geração da resposta caso ela seja muito longa.

### 🔹 Versão 3 (`v3/`) - Suporte a IA Multimodal
A terceira e última versão transforma a aplicação em uma ferramenta **Multimodal**, capaz de interpretar mais do que apenas texto.
- Evolução da interface para aceitar **anexos de arquivos**.
- Implementação de um `input` de arquivo que suporta imagens (`image/*`) e áudio (`audio/*`).
- O sistema agora permite que o usuário envie um arquivo como contexto e adicione perguntas sobre ele, demonstrando a versatilidade de modelos modernos do Chrome que conseguem fazer o processamento de visão (interpretação de imagens) e audição (processamento de fala e transcrição).

---

## Como Executar

Para executar qualquer uma das versões, basta iniciar um servidor web local dentro da pasta correspondente. O comando `npx serve .` é recomendado:

```bash
# Exemplo rodando a v3
cd v3
npx serve .
```

Em seguida, acesse `http://localhost:3000` em um navegador compatível (preferencialmente Google Chrome Canary ou versões recentes com as *flags* de Prompt API ativadas em `chrome://flags/`).
