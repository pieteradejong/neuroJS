var NeuralNet = function(layerSizes) {

  if(!Array.isArray(layerSizes) || layerSizes.length < 2) throw "Must provide array of at least two layers.";

  this._EPSILON       = .12;
  this._lambda        = .1;
  
  var weightsInitInterval = [-1, 1];
  this._layerSizes = layerSizes;
  this.initializeRandomWeights(weightsInitInterval);
  this.initializeActivations(layerSizes);
  this.initializeWeightedSums(layerSizes);
}

NeuralNet.prototype = {
  train : function (examples, options) {
    // example = {input: [2,5,3,1], output: 7}
    var options = options || {};
    var learningRate    = options.learningRate || .2;
    var errorThreshold  = options.errorThreshold || .004;
    var maxIterations   = options.maxIterations || 100;
    
    this._numExamples     = examples.length;

    var cost            = 0;
    var countIterations = 0;


    console.log("before training:\n");
    console.log("activations:", this._activations);
    console.log("weights at START: ", this._weights[0].elements[0]);
    console.log("deltas: ", this._deltas);

    
    console.log("before training:\n");
    do {
      countIterations++;
      this.initializeDeltas();
      for(var i = 0; i < this._numExamples; i++) {
        var example = examples[i];
        this.forwardProp(example.input);
        // console.log("activations at end of one forwardProp: ", this._activations[2].elements[0])

        // console.log("Activations: ", this._activations[2].elements);
        var target = this.targetToLogicalVector(example.output)
        cost += this.costFuncLogistic(this._activations[this._activations.length-1], target);
        this.backwardProp(target);
        // console.log("weight at end of one iteration: ", this._weights[1].elements[0]);
      }
      cost = cost / this._numExamples;
      // console.log("iteration number: ", countIterations, "cost: ", cost);
      // console.log("weights: ", this._weights[0].elements[0]);
      // console.log("deltas: ", this._deltas[0].elements);
      this.updateWeights(learningRate);
      // console.log("\nweight at end of one iteration: ", this._weights[0].elements[0]);
      console.log("\ndeltas at end of one iteration: ", this._deltas[1].elements);

    } while (countIterations < maxIterations && cost > errorThreshold);
    
    // debugger;
    // this.initializeActivations(this._layerSizes);
    // console.log("activations after init:", this._activations);
    console.log("weights at END: ", this._weights[0].elements[0]);
    return trainStats = {
      cost: cost,
      countIterations: countIterations
    }

  },

  forwardProp : function(example) {
    // console.log("\n3rd layer Activations before forwardProp: ", this._activations[2].elements[0]);
    // console.log("\nlayer Activations before forwardProp: ", this._activations);
    this._activations[0]    = Vector.create(example);
    this._weightedSums[0]   = Vector.create(example);
    for(var i = 1; i < this._layerSizes.length; i++) {
      this._weightedSums[i] = this._weights[i-1].multiply(this._activations[i-1]);
      this._activations[i]  = this.sigmoid(this._weightedSums[i]);
    }
    // console.log("3rd layer Activations after forwardProp: ", this._activations[2].elements[0]);
    // console.log("Activations after forwardProp: ", this._activations);
  },

  backwardProp : function(target) {
    this._deltas[this._deltas.length-1] = this._activations[this._activations.length-1].subtract(target);
    for(var i = this._deltas.length-2; i >= 0; i--) {
      this._deltas[i] = this._weights[i+1].transpose().multiply(this._deltas[i+1]);//one of these is null
      console.log("deltas during backprop:", this._deltas[1].elements);
      var factor = this.sigmoidGradient(this._weightedSums[i]);
      this._deltas[i].map(function(x,i) {
        return x * factor.e(i);
      });
    }
  },

  updateWeights : function(learningRate) {
    for(var i = 0; i < this._weights.length; i++) {
      this._weights[i].subtract(this._deltas[i].multiply(learningRate));
    }
  },

  initializeDeltas : function() {
    // debugger;
    this._deltas = Array(this._layerSizes.length-1);
    for(var i = 0; i < this._deltas.length; i++) {
      this._deltas[i] = Vector.Zero(this._layerSizes[i+1])
    }
  },


  costFuncLogistic : function(activations_last, target, weights) {
    var weights = weights || this._weights; // allow for gradient checking using +/- EPSILON
    var ones = Vector.Zero(activations_last.length).map(function(x){
      return 1;
    });

    var log_activations = activations_last.map(function(x) {
      return Math.log(x);
    });
    var term1 = -1 * target.dot(log_activations);
    var log_activations_minus1 = activations_last.map(function(x) {
      return Math.log(1-x);
    });
    var helperterm3 = target.map(function(x) {
      return 1 - x;
    });
    var term2 = -1 * helperterm3.dot(log_activations_minus1); // PROBLEM
    return term1 + term2;
  },

  sumWeights: function() {
    var totalSum = 0;
    for(var i = 0; i < this._weights.length; i++) {
      var matrix = this._weights[i];
      var cols = matrix.cols();
      for(var c = 0; c < cols; c++) {
        totalSum += matrix.col(c).each(function(x) {
          return 
        });
      }
    }
  },


  sigmoid : function(z) {
    return z.map(function(x) {
      return 1 / (1 + Math.exp(-1 * x) );
    });
  },

  sigmoidGradient : function(z) {
    return z.map(function(x) {
      return (1 / (1 + Math.exp(-1 * x) )) * (1- (1 / (1 + Math.exp(-1 * x) )));
    });
  },

  initializeRandomWeights : function(interval) {
    this._weights = Array(this._layerSizes.length - 1);
    var that = this;
    // debugger;
    for(var i = 0; i < this._weights.length; i++) {
      this._weights[i] = Matrix.Zero(this._layerSizes[i+1], this._layerSizes[i]);
      this._weights[i] = this._weights[i].map(function(x) {
        return that.randomInRange(interval[0], interval[1]);
      });
    }
  },

  randomInRange : function(min, max) {
    return Math.random() * (max - min) + min;
  },

  ones: function(size) {
    var arr = Array(size);
    for(var i = 0; i < size; i++) {
      arr[i] = 1;
    }
    return arr;
  },

  initializeActivations : function(layerSizes) {
    
    this._activations = Array(layerSizes.length);
    for(var i = 0; i < this._activations.length; i++) {
      this._activations[i] = Vector.Zero(layerSizes[i]);
    }
  },

  initializeWeightedSums : function() {
    
    this._weightedSums = Array(this._layerSizes.length);
    for(var i = 1; i < this._layerSizes.length; i++) {
      this._weightedSums[i] = Vector.Zero(this._layerSizes[i]);
    }
  },

  run : function (example) {
    this.forwardProp(example);
    return _.last(this._activations);
  },

  targetToLogicalVector: function(target) {
    if(typeof target !== 'number') throw "Argument must be number.";
    var result = Array(_.last(this._layerSizes));
    for(var i = 0; i < result.length; i++) {
      result[i] = 0;
    }
    result[target-1] = 1;
    return Vector.create(result);
  },

  // gradientChecking : function() {
  //   var plus_EPSILON  = this.costFuncLogistic.call(this, _.last(this._activations), target, nm.sub(weights, this._EPSILON))
  //   var minus_EPSILON = this.costFuncLogistic.call(this, _.last(this._activations), target, nm.add(weights, this._EPSILON));
  //   return (plus_EPSILON + minus_EPSILON) / ( 2 * this._EPSILON );
  // }

}