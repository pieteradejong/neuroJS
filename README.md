#neuroJS: Neural Network library

##About
neuroJS is a neural network library written in JavaScript.

A neural network is a predictive model often used in machine learning. It consists of nodes organized in layers, and connections between them. For example:

![neural network](http://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg)

This library can:

- Create a neural network with a given number of layers and nodes for each layer.
- Train the net on a collection of training examples.
- Feed the net previously unseen examples to predict the output.

The network will stop training when either of these conditions is reached:
- maximum number of iterations, as specified by `maxIterations`
- maximum classification error, as specified by `errorThreshold`

##Dependencies
- The Sylvester.js library was used to perform vector/matrix calculations.
- Underscore.js was used to perform array operations.

##Files:
- src/neural.js: the neural network
- lib/underscore.min.js: Underscore utility library
- lib/sylvester.js: vector/matrix/geometry library for JavaScript 
- specs/test.html: use to load neural net in the browser

##Usage 
Use the library by opening test.html in either Chrome of Firefox and opening the console.

###Create a new network:

To create a new network `net`, run this command in the console:

    var net = new NeuralNet(layerSizes[, options])

`layerSizes` (required): array with the numbers of nodes in each layer, including input and output layer;
- for example, `layerSizes = [2, 3, 1]` creates a network with 2 input nodes, one hidden layer with 3 nodes 
and an output layer with 1 node.

- `options` (optional): JavaScript object with configuration parameters for the Net:

- `costThreshold`: the network error value at which to stop training. 

- `learningRate`: the rate at which the network is updated.

- `maxIterations`: maximum number of iterations over training data set.

- `lambda`: regularization parameter that prevents overfitting.

Default settings:

    {
    costThreshold: 0.05,
    learningRate: 0.01,
    maxIterations: 500,
    lambda: 1
    }

###Train
    net.train(examples)

for example, train an 'AND' gate:

    [
    {input: [0,0], output: 0},
    {input: [0,1], output: 0},
    {input: [0,1], output: 0},
    {input: [1,1], output: 1}
    ]

### Predict
predict the output of [0,1]:

    net.predict([0,1])

## Challenges

- JavaScript is not ideal for matrix algebra.
- Switched from numeric.js library to Sylvester.js library in the middle of the project.
- Neural Nets are somewhat opaque, so debugging can be hard.

# Screenshots

This screenshot schows the slight change in network parameters after each iteration, as they are updated
to better fit the input and output for which it's been trained.

Also shown is the cost after each iteration. The cost decreases as the network learns the desired output.


## Starting training

![neural network](http://pieteronline.com/neurojs/specs/training_start.png)

## Finish training; prediction

After training, the network makes a prediction given an input we give it.
In this screenshot, it is predicting the ouput of an `OR` gate with input `[0,1]`.
The network approximates the actual output, which would be 1.

![neural network](http://pieteronline.com/neurojs/specs/training_prediction.png)
