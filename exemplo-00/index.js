import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada da rede:
    // inputShape: número de características de entrada
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    // units: número de neurônios na camada (80)
    // grande quantidade de neurônios devido a pequena base de treino
    // quanto mais neurônios, maior a capacidade de aprendizado
    // mas também maior o risco de overfitting
    // activation: função de ativação (relu)
    // relu: função de ativação que retorna o valor de entrada se for positivo, e 0 se for negativo
    // é a função de ativação mais comum em redes neurais

    model.add(tf.layers.dense({ units: 80, inputShape: [7], activation: 'relu' }));

    // saida: 3 neuronios
    // softmax: função de ativação que retorna a probabilidade normalizada de cada categoria
    // premium, medium, basic

    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilação do modelo
    // optimizer: Adam (Adaptative Moment Estimation). Algoritmo usado para ajustar os pesos da rede de forma eficiente e inteligente
    // Aprende com histórico de erros e acertos

    // loss: função de perda usada para medir o erro do modelo
    // Compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // ex: a categoria premium será sempre [1, 0, 0]
    // o modelo vai tentar se aproximar o máximo possível dessa resposta
    // quanto mais próximo, menor a perda
    // quanto mais distante a previsão for da resposta correta, maior a perda
    // Exemplo clássico: categorização de imagens, textos e usuários, recomendações
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"

    // metrics: métricas usadas para avaliar o desempenho do modelo
    // accuracy: acurácia. porcentagem de acertos

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamento do modelo
    await model.fit(inputXs, outputYs, {
        verbose: 0, // 0 = não mostra nada, 1 = mostra barra de progresso, 2 = mostra logs
        epochs: 100, // número de vezes que o modelo vai ver o dataset
        shuffle: true, // embaralha os dados a cada época para evitar que o modelo memorize a ordem
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
                // console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.accuracy}`);
            }
        }
    })

    return model;

}

async function predict(model, pessoaNormalizada) {
    const tensorPessoa = tf.tensor2d([pessoaNormalizada]);
    const predictions = model.predict(tensorPessoa);
    const predArray = await predictions.array();
    return predArray[0].map((prob, index) => ({ prob, index }));
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();


// quanto mais, melhor!
// assim o algoritmo consegue entender melhor os padrões complexos dos dados
const model = await trainModel(inputXs, outputYs);

const pessoa = {
    nome: "Victor",
    idade: 28,
    cor: "vermelho",
    localizacao: "São Paulo"
}

// Normaliza a idade da pessoa para o modelo entender
// exemplo: idade_min = 25, idade_max = 40, 
// então idade_normalizada = (28 - 25) / (40 - 25) = 0.2

const pessoaNormalizada = [
    0.2,
    pessoa.cor === "azul" ? 1 : 0,
    pessoa.cor === "vermelho" ? 1 : 0,
    pessoa.cor === "verde" ? 1 : 0,
    pessoa.localizacao === "São Paulo" ? 1 : 0,
    pessoa.localizacao === "Rio" ? 1 : 0,
    pessoa.localizacao === "Curitiba" ? 1 : 0
];

const predictions = await predict(model, pessoaNormalizada);
const resultado = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
    .join("\n");


console.log(resultado);