import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const pesos = {
    age: 0.1,
    price: 0.2,
    color: 0.3,
    category: 0.4
}

const normalize = (value, min, max) => (value - min) / (max - min) || 1;

function makeContext(catalog, users) {
    const ages = users.map(u => u.age);
    const maxAge = Math.max(...ages);
    const minAge = Math.min(...ages);

    const prices = catalog.map(p => p.price);
    const maxPrice = Math.max(...prices);
    const minPrice = Math.min(...prices);

    const colors = [...new Set(catalog.map(p => p.color))];
    const categories = [...new Set(catalog.map(p => p.category))];

    const colorIndex = Object.fromEntries(colors.map((color, index) => [color, index]));
    const categoryIndex = Object.fromEntries(categories.map((category, index) => [category, index]));

    const midAge = (maxAge + minAge) / 2;
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    const midAgePerProduct = Object.fromEntries(
        catalog.map(product => {
            const avgAge = ageCounts[product.name] ? ageSums[product.name] / ageCounts[product.name] : midAge;
            return [product.name, normalize(avgAge, minAge, maxAge)];
        })
    );

    return {
        catalog,
        colorIndex,
        categoryIndex,
        midAgePerProduct,
        midAge,
        maxAge,
        minAge,
        maxPrice,
        minPrice,
        numColors: colors.length,
        numCategories: categories.length,
        users,
        dimentions: 2 + colors.length + categories.length // age, price, color, category
    }
}

const oneHotWeighted = (index, lenght, weight) =>
    tf.oneHot(index, lenght).cast('float32').mul(weight);

function encodeProduct(p, ctx) {
    const price = tf.tensor1d([normalize(p.price, ctx.minPrice, ctx.maxPrice) * pesos.price]);
    const age = tf.tensor1d([ctx.midAgePerProduct[p.name] ?? 0.5 * pesos.age]);

    const category = oneHotWeighted(ctx.categoryIndex[p.category], ctx.numCategories, pesos.category);
    const color = oneHotWeighted(ctx.colorIndex[p.color], ctx.numColors, pesos.color);

    return tf.concat([price, age, color, category]);
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('/data/products.json')).json();
    const ctx = makeContext(catalog, users);

    ctx.productVectors = catalog.map(p => {
        return {
            name: p.name,
            meta: { ...p },
            vector: encodeProduct(p, ctx).dataSync()
        }
    })
    debugger

    _globalCtx = ctx;
    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
