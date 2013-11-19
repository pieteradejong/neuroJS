var NeuralNet = function(layerSizes) {
  if(!Array.isArray(layerSizes) || layerSizes.length < 2) {
    throw "Must provide array of at least two layers.";
  }  
  this._layerSizes = layerSizes;
  
  var interval = [-.5 , .5];
  this.initializeRandomWeights(interval);
}

NeuralNet.prototype = {
  train: function(examples, options) {
    var deltas, cost, input, target, activations, errors, deltasUpdate,
        options, costThreshold, learningRate, maxIterations, lambda, countIterations;

    options         = options                || {};
    costThreshold   = options.costThreshold  ||    .005;
    learningRate    = options.learningRate   ||    .05;
    maxIterations   = options.maxIterations  || 500;
    lambda          = options.lambda         ||   1;
    
    countIterations = 0;
    
    console.log("Started training...", "\n");

    do {
      countIterations++
      deltas = this.initializeDeltas();
      cost = 0

      for(var i = 0; i < examples.length; i++) {
        input          = Vector.create(examples[i].input);
        target         = this.integerToBinaryVector(examples[i].output);
        activations    = this.forwardProp(input);
        cost          += this.computeCostLogistic(_.last(activations), target);
        errors         = this.computeErrors(activations, target);
        deltasUpdate   = this.computeDeltas(activations, errors);
        
        for(var k = 0; k < deltas.length; k++) {
          deltas[k] = deltas[k].add(deltasUpdate[k]);
        }
      }
      
      for(var i = 0; i < deltas.length; i++) {
        var weights_i = this._weights[i];
        deltas[i] = deltas[i].map(function(x, row, col) {
          return x * (1/examples.length) + lambda/examples.length * weights_i.e(row, col);
        });
      }

      for(var k = 0; k < this._weights.length; k++) {
        var deltas_k = deltas[k].multiply(learningRate);
        this._weights[k] = this._weights[k].subtract(deltas_k);
      }
    
      cost *= -1 / examples.length;

      for(var k = 0; k < this._weights.length; k++) {
        var matrixSquared = this._weights[k].map(function(x) {
          return Math.pow(x, 2);
        });
        cost += lambda/(2*examples.length) * this.sumMatrix(matrixSquared);
      }

      console.log("Iteration number: ", countIterations, " cost: ", cost);

    } while (cost > costThreshold && countIterations < maxIterations);
    
    console.log("\n", "Training complete after ", countIterations, " iterations; Cost: ", cost, "\n");

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

    cost += result.map(function(x) {
      return Math.log(x);
    }).dot(target);
    
    cost += target.map(function(x) {
      return 1-x;
    }).dot(result.map(function(x){
      return Math.log(1-x);
    }));

    return cost;
  },
    
  computeErrors: function(activations, target) {
    var errors = [];

    errors[this._layerSizes.length-2] = _.last(activations).subtract(target);
    
    for(var i = this._layerSizes.length - 3; i >= 0; i--) {
      errors[i] = (this._weights[i+1].transpose()).multiply(errors[i+1]);
      var sigmoidDeriv = activations[i+1].map(function(x) {
        return x * (1 - x);
      });
      errors[i] = errors[i].map(function(x, k) {
        return x * sigmoidDeriv.e(k);
      });
    }

    return errors;
  },

  computeDeltas: function(activations, errors) {
    var deltasUpdate = [];

    for(var i = 0; i < this._layerSizes.length - 1; i++) {
      var n = Matrix.create(errors[i].elements);
      var m = Matrix.create(activations[i].elements).transpose();
      deltasUpdate[i] = n.multiply(m);
    }

    return deltasUpdate;
  },

  initializeRandomWeights: function(interval) {
    this._weights = [];

    var that = this;
    for(var i = 0; i < this._layerSizes.length - 1; i++) {
      this._weights[i] = Matrix.Zero(this._layerSizes[i+1], this._layerSizes[i]);
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

  sumMatrix: function(matrix) {
    var rows = matrix.rows();
    var sum = 0;

    for(var i = 1; i <= rows; i++) {
        sum += _.reduce(matrix.row(i).elements, function(sum, number){
        return sum + number;
      }, 0);
    }

    return sum;
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
