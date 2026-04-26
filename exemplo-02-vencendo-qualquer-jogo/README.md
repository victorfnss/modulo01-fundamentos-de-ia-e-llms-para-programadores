# Duck Hunt JS - AI Bot Edition (Exemplo 02)

This is a modified version of the classic DuckHunt game implemented in Javascript and HTML5. In this specific project, we integrated an **Artificial Intelligence Bot** using **TensorFlow.js** and the **YOLOv5** object detection model to play the game autonomously!

## Integrated Artificial Intelligence 🧠🎯

Through the Web Workers API (`machine-learning/worker.js`), the game captures the screen canvas, processes the pixels using tensors, and runs inferences on the local YOLOv5 model. 
When the model identifies the targets, it calculates the coordinates and sends automatic "shots" back to the main game thread, effectively "beating any game" without human interaction.

## Core Technologies

- **Machine Learning**: TensorFlow.js and YOLOv5n (Web Model).
- **Rendering**: Supports WebGL and Canvas via the PixiJS rendering engine.
- **Audio**: Uses WebAudioAPI and fallbacks to HTML5 Audio via HowlerJS.
- **Game Logic**: Managed using ES6 classes transpiled to ES5 using Babel.

## Working With This Repo

 - You must have [nodejs](https://nodejs.org/) installed.
 - Clone the repo into a directory of your choice
 - `cd` into that directory and run `npm install`
 - Use `npm start` to start a local webserver which will make the site available at `http://localhost:8080/`. Cross origin errors prevent this project from being accessed in the browser with the `file://` protocol. This will also trigger automatic builds and reloads of the page when changes are detected in the `src` directory.
 - If you want to manually cut a build of the application code run `npm run build`
 
## Working With Audio and Visual Assets

This repo ships with committed dist files to make it easy for developers to get up and running. If you want to customize the way this game looks and sounds, you'll need to work with audio and image sprites. The following tasks make that possible: 

 - To rebuild audio assets use `npm run audio` (there is a hard dependency on [ffmpeg](https://ffmpeg.org/download.html) to run this task)
 - To rebuild image assets use `npm run images` (there is a hard dependency on [texturepacker](https://www.codeandweb.com/texturepacker/download) to run this task)

## Bugs
Please report bugs as [issues](https://github.com/MattSurabian/DuckHunt-JS/issues).

## Contributing
Pull requests are welcome! Please ensure code style and quality compliance with `npm run lint` and include any built files.
