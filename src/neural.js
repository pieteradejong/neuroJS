var NeuralNet = function(layerSizes) {

  if(!Array.isArray(layerSizes) || layerSizes.length < 2) throw "Must provide array of at least two layers.";

  this._EPSILON       = .12;
  this._lambda        = .1;
  
  var weightsInitInterval = [-.01, .01];
  this._layerSizes = layerSizes;
  this.initializeRandomWeights(layerSizes, weightsInitInterval);
  this.initializeActivations(layerSizes);
  this.initializeWeightedSums(layerSizes);
}

NeuralNet.prototype = {
  train : function (examples, options) {
    // example = {input: [2,5,3,1], output: 7}
    var options = options || {};
    var learningRate    = options.learningRate || .1;
    var errorThreshold  = options.errorThreshold || .004;
    var maxIterations   = options.maxIterations || 100;
    
    this._numExamples     = examples.length;

    var cost            = 0;
    var countIterations = 0;

    do {
      countIterations++;
      for(var i = 0; i < this._numExamples; i++) {
        var example = examples[i];
        this.forwardProp(example.input);
        cost += this.costFuncLogistic(_.last(this._activations), example.output);
        this.backwardProp(example.output);
      }
      cost = cost / this._numExamples;
      console.log("iteration number: ", countIterations, "cost: ", cost);
      this.updateWeights(learningRate);
    } while (countIterations < maxIterations && cost > errorThreshold);
    
    return trainStats = {
      cost: cost,
      countIterations: countIterations
    }
  },

  forwardProp : function(example) {
    this._activations[0] = example;
    this._weightedSums[0] = example;
    for(var i = 1; i < this._layerSizes.length; i++) {
      this._weightedSums[i] = nm.dotMV(this._weights[i-1], this._activations[i-1]);
      this._activations[i] = this.sigmoid(this._weightedSums[i]);
    }
  },

  backwardProp : function(target) {
    // console.log("deltas before init: ", this._deltas); // undefined
    this.initializeDeltas();
    // console.log("deltas after init: ", this._deltas[0], this._deltas[1]); // [0,0][0,0,0] OK
    // check components defined?
    // console.log("activations: ", this._activations); // OK
    // console.log("last layer normal activations: ", this._activations[this._activations.length-1]);
    // console.log("last layer activations: ", _.last(this._activations)); 
    // console.log("target: ", target);
    this._deltas[this._deltas.length-1] = nm.sub(this._activations[this._activations.length-1], target);
    // break;
    // console.log("deltas at init last layer: ", this._deltas);
    for(var i = this._deltas.length-2; i >= 0; i--) {
      // console.log("weights and deltas: ", this._weights[i], this._deltas[i+1]);
      this._deltas[i] = nm.mul(nm.dotMV(this._weights[i], this._deltas[i+1]), this.sigmoidGradient(this._weightedSums[i]));
      // console.log("deltas: ", this._deltas);
    }
  },

  costFuncLogistic : function(activations_last, target, weights) {
    var weights = weights || this._weights; // allow for gradient checking using +/- EPSILON
    var ones = this.ones(activations_last.length);
    var term1 = -1 * nm.mul(target, nm.log(_.last(this._activations)));
    var term2 = -1 * nm.mul(nm.sub(ones, target), nm.log(nm.sub(ones, _.last(this._activations))));
    var regularization = this._lambda/(2*this._numExamples) * _.reduce(this._weights, function(memo, value) {
      return memo + nm.sum(value);
    }, 0);
    return term1 + term2 + regularization;
  },

  updateWeights : function(learningRate) {
    for(var i = 0; i < this._weights.length; i++) {
        nm.sub(this._weights[i], nm.mul(learningRate, this._deltas[i]));
      }
  },

  initializeDeltas : function() {
    this._deltas = Array(this._layerSizes.length - 1);
    for(var i = 0; i < this._deltas.length; i++) {
      this._deltas[i] = this.zeros(this._layerSizes[i+1]);
    }
    console.log("verify deltas corectly initialized: ", this._deltas);
  },

  sigmoid : function(z) {
    if(!Array.isArray(z)) throw "Input must be array."
    var result = nm.neg(z);
    result = nm.exp(result);
    result = nm.addVS(result, 1);
    result = nm.div(1, result);
    return result;
  },

  sigmoidGradient : function(z) {
    return nm.mul(this.sigmoid(z), nm.addVS(nm.neg(this.sigmoid(z)), 1));
  },

  initializeRandomWeights : function(interval) {
    this._weights = [];
    for(var i = 0; i < this._layerSizes.length - 1; i++) {
      this._weights[i] = [];
      for(var j = 0; j < this._layerSizes[i+1]; j++) {
        this._weights[i][j] = [];
        for(var k = 0; k < this._layerSizes[i]; k++) {
          this._weights[i][j][k] = this.randomInRange(interval[0], interval[1]);
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

  zeros: function(size) {
    var arr = Array(size);
    for(var i = 0; i < size; i++) {
      arr[i] = 0;
    }
    return arr;
  },

  initializeActivations : function(layerSizes) {
    this._activations = Array(layerSizes.length);
    for(var i = 0; i < this._activations.length; i++) {
      this._activations[i] = this.zeros(layerSizes[i]);
    }
    // break;
    // console.log("verify activations corectly initialized: ", this._activations[0],this._activations[1],this._activations[2] );
    // [0,0,0][0,0][0,0,0] OK
  },

  initializeWeightedSums : function() {
    this._weightedSums = [];
    for(var i = 1; i < this._layerSizes.length; i++) {
      this._weightedSums[i] = [];
      for(var j = 0; j < this._layerSizes.length[i]; j++) {
        this._weightedSums[i][j] = 0;
      }
    }
  },

  run : function (example) {
    this.forwardProp(example);
    return _.last(this._activations);
  },

  gradientChecking : function() {
    var plus_EPSILON  = this.costFuncLogistic.call(this, _.last(this._activations), target, nm.sub(weights, this._EPSILON))
    var minus_EPSILON = this.costFuncLogistic.call(this, _.last(this._activations), target, nm.add(weights, this._EPSILON));
    return (plus_EPSILON + minus_EPSILON) / ( 2 * this._EPSILON );
  }

}