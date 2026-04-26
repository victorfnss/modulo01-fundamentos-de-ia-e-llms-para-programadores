# E-commerce Recommendation System

A web application that displays user profiles and product listings, applying a **machine learning recommendation engine using TensorFlow.js**. It tracks user purchases to dynamically build recommendations tailored to their profiles.

## Project Structure

- `index.html` - Main HTML file for the application
- `index.js` - Entry point for the application
- `view/` - Contains classes for managing the DOM and templates
- `controller/` - Contains controllers to connect views and services
- `service/` - Contains business logic for data handling
- `data/` - Contains JSON files with user and product data
- `workers/` - Contains Web Workers (like `modelTrainingWorker.js`) that handle heavy ML operations off the main thread.

## Setup and Run

1. Install dependencies:
```bash
npm install
```

2. Start the application:
```bash
npm start
```

3. Open your browser and navigate to `http://localhost:8080`

## Features

- User profile selection with details display
- Past purchase history display
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage
- **Neural Network Training:** Uses TensorFlow.js in a Web Worker to encode products and users based on attributes (price, category, color, age).
- **Dynamic Product Recommendations:** Calculates prediction scores based on the trained model and sorts products to recommend the most relevant ones.