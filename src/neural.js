var NeuralNet = function(layerSizes) {

  if(!Array.isArray(layerSizes) || layerSizes.length < 2) throw "Must provide array of at least two layers.";

  this._EPSILON       = .12;
  this._lambda        = .2;
  this._weight_init_min = -.5;
  this._weight_init_max = .5;
  
  this.layerSizes = layerSizes;
  this._numLayers = layerSizes.length;
  this.initializeRandomWeights(layerSizes);
  this.initializeActivations(layerSizes);
  this.initializeWeightedSums(layerSizes);
}

NeuralNet.prototype = {
  train : function (examples, options) {
    // example = {input: [2,5,3,1], output: 7}
    console.log("training...");
    var options = options || {};
    var learningRate    = options.learningRate || .2;
    var errorThreshold  = options.errorThreshold || .5;
    var maxIterations   = options.maxIterations || 100;
    
    this._numExamples     = examples.length;

    var cost            = 0;
    var countIterations = 0;

    do {
      countIterations++;
      console.log(countIterations);
      for(var i = 0; i < this._numExamples; i++) {
        var example = examples[i];
        this.forwardProp(example.input);
        cost += this.costFuncLogistic(_.last(this._activations), example.output);
        this.backwardProp(example.output);
      }
      this.updateWeights(learningRate);
    } while (countIterations <= maxIterations && cost > errorThreshold);
    
    return trainStats = {
      cost: cost,
      countIterations: countIterations
    }
  },

  forwardProp : function(example) {
    console.log("forwardProp...");
    this._activations[0] = example;
    this._weightedSums[0] = example;
    for(var i = 1; i < this._numLayers; i++) {
      this._weightedSums[i] = numeric.dotMV(this._weights[i-1], this._activations[i-1]);
      this._activations[i] = this.sigmoid(this._weightedSums[i]);
    }
  },

  backwardProp : function(target) {
    console.log("backwardProp...");
    this.initializeDeltas();
    this._deltas[this._numLayers] = numeric.sub(this._activations[this._numLayers], target);
    for(var i = this._numLayers - 1; i >= 1; i--) {
      this._deltas[i] = numeric.mul(numeric.dot(this._weights[i], this._deltas[i+1]), this.sigmoidGradient(this._weightedSums[i]));
    }
  },

  costFuncLogistic : function(activations_last, target, weights) {
    // compute cost for one example, given activation vector and target vector, and potentially weights
    var weights = weights || this._weights; // allow for gradient checking using +/- EPSILON
    var ones = this.ones(activations_last.length);
    var term1 = -1 * numeric.mul(target, numeric.log(_.last(this._activations)));
    var mul1 = numeric.sub(ones, target); // OK
    var mul2 = numeric.log(numeric.sub(ones, _.last(this._activations))); // NOT OK
    var term2 = -1 * numeric.mul(mul1,mul2);
    var regularization = this._lambda / ( 2 * this._numExamples );
    var regularization = regularization * _.reduce(this._weights, function(memo, value) {
      return memo + numeric.sum(value);
    }, 0);
    return term1 + term2 + regularization;
  },

  updateWeights : function(learningRate) {
    console.log("updating weights...");
    for(var i = 0; i < this._weights.length; i++) {
        numeric.sub(this._weights[i], numeric.mul(learningRate, this._deltas[i]));
      }
  },

  sigmoid : function(z) {
    if(!Array.isArray(z)) throw "Input must be array."
    var result = numeric.neg(z);
    result = numeric.exp(result);
    result = numeric.addVS(result, 1);
    result = numeric.div(1, result);
    return result;
  },

  sigmoidGradient : function(z) {
    return numeric.mul(this.sigmoid(z), numeric.addVS(numeric.neg(this.sigmoid(z)), 1));
  },

  initializeRandomWeights : function() {
    this._weights = [];
    for(var i = 0; i < this.layerSizes.length - 1; i++) {
      this._weights[i] = [];
      for(var j = 0; j < this.layerSizes[i+1]; j++) {
        this._weights[i][j] = [];
        for(var k = 0; k < this.layerSizes[i]; k++) {
          this._weights[i][j][k] = this.randomInRange(-.5, .5);
        }
      }
    }
  },

  randomInRange : function (min, max) {
    return Math.random() * (max - min) + min;
  },

  ones: function(size) {
    var arr = Array(size);
    for(var i = 0; i < size; i++) {
      arr[i] = 1;
    }
    return arr;
  },

  initializeDeltas : function() {
    this._deltas = [];
    for(var i = 1; i < this.layerSizes.length; i++) {
      this._deltas[i] = [];
      for(var j = 0; j < this.layerSizes[i]; j++) {
        this._deltas[i][j] = 0;
      }
    }
  },

  initializeActivations : function() {
    this._activations = [];
    for(var i = 0; i < this.layerSizes.length; i++) {
      this._activations[i] = [];
      for(var j = 0; j < this.layerSizes[i]; j++) {
        this._activations[i][j] = 0;
      }
    }
  },

  initializeWeightedSums : function() {
    this._weightedSums = [];
    for(var i = 1; i < this._numLayers; i++) {
      this._weightedSums[i] = [];
      for(var j = 0; j < this._numLayers[i]; j++) {
        this._weightedSums[i][j] = 0;
      }
    }
  },

  run : function (example) {
    this.forwardProp(example);
    return _.last(this._activations);
  },

  gradientChecking : function() {
    var plus_EPSILON  = this.costFuncLogistic.call(this, _.last(this._activations), target, numeric.sub(weights, this._EPSILON))
    var minus_EPSILON = this.costFuncLogistic.call(this, _.last(this._activations), target, numeric.add(weights, this._EPSILON));
    return (plus_EPSILON + minus_EPSILON) / ( 2 * this._EPSILON );
  }

}