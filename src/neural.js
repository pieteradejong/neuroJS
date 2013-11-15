var NeuralNet = function(layerSizes) {
  if(!Array.isArray(layerSizes) || layerSizes.length < 2) {
    throw "Must provide array of at least two layers.";
  }
  // set instance properties:
  this._layerSizes = layerSizes;
  
  var interval = [-.5 , .5];
  this.initializeRandomWeights(interval);
}

NeuralNet.prototype = {
  train: function(examples, options) {
    // data input format: array of objects (examples): [{input: [1,8,3,5,22], output: 5}, ...]
    var options = options || {};
    var costThreshold = options.costThreshold || .005;
    var learningRate = options.learningRate || .2;
    var maxIterations = options.maxIterations || 20000;
    var lambda = options.lambda || 0;
    
    var countIterations = 0
    
    do {
      // console.log("\n\nbegin new iteration", countIterations);
      countIterations++
      // deltas: initialize to zeros:
      var deltas = this.initializeDeltas();
      var cost = 0
      for(var i = 0; i < examples.length; i++) {
        // vectorize input and target
        var input       = Vector.create(examples[i].input);
        var target      = this.integerToBinaryVector(examples[i].output);
        var activations = this.forwardProp(input); // result is array of Vectors
        cost           += this.computeCostLogistic(_.last(activations), target); // cost is number
        // use errors to compute Deltas
        var errors       = this.computeErrors(activations, target);
        
        var deltasUpdate = this.computeDeltas(activations, errors);
        for(var k = 0; k < deltas.length; k++) {
          deltas[k] = deltas[k].add(deltasUpdate[k]);
        }
      }
      // divide by number of examples
      for(var i = 0; i < deltas.length; i++) {
        deltas[i] = deltas[i].map(function(x) {
          return x * (1/examples.length); // + regularization
        });
      }
      for(var k = 0; k < this._weights.length; k++) {
        this._weights[k] = this._weights[k].subtract(deltas[k].multiply(learningRate));
      }
      cost = cost * -1 / examples.length; // + regularization

      if(countIterations % 100 === 0) {
        console.log("end of new iteration, cost:" , cost);
      }
    } while (cost > costThreshold && countIterations <= maxIterations);
    return {
      cost: cost,
      iterations: countIterations
    }
  },

  forwardProp: function(example) {    
    var activations = [];
    activations[0] = example;
    for(var i = 1; i < this._layerSizes.length; i++) {
      activations[i] = this._weights[i-1].multiply(activations[i-1]);
      activations[i] = activations[i].map(function(x) {
        return 1/(1+Math.exp(-1*x));
      });
    }
    return activations;
  },
    
  predict: function(example) {
    var activations = this.forwardProp(Vector.create(example));
    return activations[activations.length-1];
  },

  computeCostLogistic: function(result, target) {
    var cost = 0;
    var term1 = result.map(function(x) {
      return Math.log(x);
    }).dot(target);
    cost += term1;
    var term2 = target.map(function(x) {
      return 1-x;
    });

    cost += term2.dot(result.map(function(x){
      return Math.log(1-x);
    }));
    return cost;
  },
    
  computeErrors: function(activations, target) {
    // activations (array of Vectors), target (Vector) -> array of Vectors
    var errors = [];
    errors[this._layerSizes.length-2] = _.last(activations).subtract(target);
    for(var i = this._layerSizes.length - 3; i >= 0; i--) {
      errors[i] = this._weights[i+1].transpose().multiply(errors[i+1]);
      var sigmoidDeriv = activations[i].map(function(x) {
        return x - (1 - x);
      });
      
      errors[i] = errors[i].map(function(x, k) {
        return x * sigmoidDeriv.e(k);
      });
    }
    return errors;
  },

  computeDeltas: function(activations, errors) {
    // activations (array of Vectors), errors (array of Vectors) -> array of Matrices
    var deltaUpdate = [];
    for(var i = 0; i < this._layerSizes.length - 1; i++) {
      var m = Matrix.create(activations[i].elements).transpose();
      var n = Matrix.create(errors[i].elements);
      deltaUpdate[i] = n.multiply(m);
    }
    return deltaUpdate;
  },

  initializeRandomWeights: function(interval) {
    this._weights = [];
    var that = this;
    for(var i = 0; i < this._layerSizes.length - 1; i++) {
      this._weights[i] = Matrix.Zero(this._layerSizes[i+1], this._layerSizes[i]); // undefined
      this._weights[i] = this._weights[i].map(function(x) {
        return that.randomInRange(interval[0], interval[1]);
      });
    }
  },

  initializeDeltas: function() {
   var deltas = [];
    for(var i = 0; i < this._layerSizes.length - 1; i++) {
      deltas[i] = Matrix.Zero(this._layerSizes[i+1], this._layerSizes[i]);
    }
    return deltas;
  },

  randomInRange: function(min, max) {
    return Math.random() * (max - min) + min;
  },

  integerToBinaryVector: function(integer) {
    var array = [];
    for(var i = 0; i < _.last(this._layerSizes); i++) {
      if(i === integer - 1) {
        array.push(1);
      } else {
        array.push(0);
      }
    }
    return Vector.create(array);
  }
};
