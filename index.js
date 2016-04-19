
var idCount = 0;

var node = function() {
  return {
    id: idCount ++,
    inputs: [],
    inputWeights: [],
    output: 0,
    error: 0,
    outputLinks: [],
    getMatchingWeightForNode: function(node) {
      for(var index = 0;index < this.inputs.length;index ++) {
        if (this.inputs[index] == node) {
          return this.inputWeights[index];
        }
      }
    },
    linkToInput: function(node) {
      this.inputs.push(node);
      this.inputWeights.push(1);
      node.linkOutput(this);
    },
    linkOutput: function(node) {
      this.outputLinks.push(node);
    },
    forward: function() {
      var val = 0;
      for (var index = 0;index < this.inputs.length;index ++) {
        val += this.inputs[index].output * this.inputWeights[index];
      }

      this.output = 1.0 / (1.0 + Math.exp(-val));
    },
    backward: function(learnRate) {
      var error = 0;
      for (var outputLinkIndex = 0;outputLinkIndex < this.outputLinks.length;outputLinkIndex ++) {
        var n = this.outputLinks[outputLinkIndex].getMatchingWeightForNode(this);
        error += this.outputLinks[outputLinkIndex].error * n;
      }

      this.error = this.output * ((1 - this.output) * error);

      for (var weightIndex = 0;weightIndex < this.inputs.length;weightIndex ++) {
        var tweakAmount = this.error * this.inputs[weightIndex].output;
        tweakAmount *= learnRate;
        this.inputWeights[weightIndex] += tweakAmount;
      }
    },
    backwardWithExpectedOutput: function(learnRate, expectedOutput) {
      this.error = this.output * ((1 - this.output) * (expectedOutput - this.output));
      for (var weightIndex = 0;weightIndex < this.inputs.length;weightIndex ++) {
        var tweakAmount = this.error * this.inputs[weightIndex].output;
        tweakAmount *= learnRate;
        this.inputWeights[weightIndex] += tweakAmount;
      }
    },
    getWeightCount: function() {
      return this.inputWeights.length;
    },
    getWeight: function(weightIndex) {
      return this.inputWeights[weightIndex];
    },
    getOutput: function() {
      return this.output;
    }
  }
};

var inputNode = function() {
  return {
    output: 0,
    outputLinks: [],
    setOutput: function(value) {
      this.output = value;
    },
    linkOutput: function(node) {
      this.outputLinks.push(node);
    }
  }
};

var layer = function() {
  return {
    nodes: [],
    initialize: function(layerSize, isInput) {
      this.nodes = [];
      for (var index = 0;index < layerSize;index ++) {
        if (isInput) {
          this.nodes.push(new inputNode());
        } else {
          this.nodes.push(new node());
        }
      }
    },
    linkOutputToInputs: function(layer) {
      // We assume the same number of nodes in the layer
      var maxNumLinks = 2; // TODO: Hard code the number of links to each node for now
      for (var index = 0;index < this.nodes.length;index ++) {
        for (var nodeLinkIndex = 0;nodeLinkIndex < maxNumLinks;nodeLinkIndex ++) {
          layer.nodes[index].linkToInput(this.nodes[(index + nodeLinkIndex) % this.nodes.length]);
        }
      }
    },
    forward: function() {
      for (var index = 0;index < this.nodes.length;index ++) {
        this.nodes[index].forward();
      }
    },
    backward: function(nodeIndex, learnRate) {
      this.nodes[nodeIndex].backward(learnRate);
    },
    backwardOutputLayer: function(nodeIndex, learnRate, expectedOutput) {
      this.nodes[nodeIndex].backwardWithExpectedOutput(learnRate, expectedOutput);
    },
    setNodeOutput: function(nodeIndex, value) {
      this.nodes[nodeIndex].setOutput(value);
    },
    getNodeOutput: function(nodeIndex) {
      return this.nodes[nodeIndex].getOutput();
    },
    getNodeCount: function() {
      return this.nodes.length;
    },
    displayToConsole: function() {
      var out = 'inputs: ';

      for (var index = 0;index < this.nodes.length;index ++) {
        for (var inputIndex = 0;inputIndex < this.nodes[index].inputs.length;inputIndex ++) {
          out += this.nodes[index].inputs[inputIndex].output.toString() + ' ';
        }
        out += ',';
      }
      console.log(out);

      out = 'weights:';
      for (var index = 0;index < this.nodes.length;index ++) {
        for (var weightIndex = 0;weightIndex < this.nodes[index].getWeightCount();weightIndex ++) {
          out += this.nodes[index].getWeight(weightIndex).toString() + ' ';
        }
        out += ','
      }
      console.log(out);

      out = 'error:  ';
      for (var index = 0;index < this.nodes.length;index ++) {
        out += this.nodes[index].error.toString() + ',';
      }
      console.log(out);

      out = 'output: ';
      for (var index = 0;index < this.nodes.length;index ++) {
        out += this.nodes[index].output.toString() + ',';
      }
      console.log(out);
    }
  }
};

var network = function() {
  return {
    trainingDataSet: [
      {
        input: [0.1, 0, 0],
        expectedOutput: [1, 0, 0]
      },
      {
        input: [0.2, 0, 0],
        expectedOutput: [0, 1, 0]
      },
      {
        input: [0.3, 0.3, 0.3],
        expectedOutput: [0, 0, 1]
      }
    ],
    layers: [],
    initialize: function(numOfLayers, layerDepth) {
      // Create the layers
      this.layers = [];
      for (var index = 0;index < numOfLayers + 1;index ++) {
        var l = new layer();
        l.initialize(layerDepth, index === 0);
        this.layers.push(l);
      }

      // link the layers
      for (var index = 0;index < numOfLayers;index ++) {
        this.layers[index].linkOutputToInputs(this.layers[index + 1]);
      }
    },
    getTrainingValues: function(trainingIteration) {
      return this.trainingDataSet[trainingIteration % this.trainingDataSet.length];
    },
    train: function(maxTrainingIterations) {
      var learnRate = 0.01;
      var trainingIteration = 0;
      var displayInterval = 100000; // Dump the network to console every n training intervals
      var trainingValuesIteration = 0;

      var inputLayer = this.layers[0];

      // the train loop
      while (trainingIteration < maxTrainingIterations) {

        // Get training data for this iteration
        var trainingValues = this.getTrainingValues(trainingValuesIteration);
        trainingValuesIteration ++;

        for (var nodeIndex = 0;nodeIndex < inputLayer.getNodeCount();nodeIndex ++) {
          inputLayer.setNodeOutput(nodeIndex, trainingValues.input[nodeIndex]);
        }

        this.forward();
        if (trainingIteration === 0) {
          this.displayToConsole();
        }

        for (var layerIndex = this.layers.length - 1;layerIndex > 0;layerIndex --) {
          var layer = this.layers[layerIndex];
          for (var nodeIndex = 0;nodeIndex < layer.nodes.length;nodeIndex ++) {
            if (layerIndex === this.layers.length - 1) {
              layer.backwardOutputLayer(nodeIndex, learnRate, trainingValues.expectedOutput[nodeIndex]);
            } else {
              layer.backward(nodeIndex, learnRate);
            }
          }
        }

        if (trainingIteration % displayInterval == 0) {
          console.log('iteration: ' + trainingIteration);
          this.displayToConsole();
          this.evaluateError(trainingValues);
        }

        trainingIteration ++;
      }

      this.displayToConsole();
    },
    evaluateError: function(trainingValues) {
      console.log('error evaluation (closer to zero = better): ');

      var outputLayer = this.layers[this.layers.length - 1];
      var errorOut = '';
      for (var nodeIndex = 0;nodeIndex < outputLayer.getNodeCount();nodeIndex ++) {
        errorOut += trainingValues.expectedOutput[nodeIndex] - outputLayer.getNodeOutput(nodeIndex) + ',';
      }

      console.log(errorOut);
      console.log('');
    },
    testAgainstTrainingData: function() {
      console.log('');
      console.log('Testing against training data: ');
      console.log('');

      var inputLayer = this.layers[0];
      var outputLayer = this.layers[this.layers.length - 1];

      for (var index = 0;index < this.trainingDataSet.length;index ++) {
        var trainingValues = this.getTrainingValues(index);

        var out = 'Test input: ';
        for (var trainingValuesIndex = 0;trainingValuesIndex < trainingValues.input.length;trainingValuesIndex ++) {
          out += trainingValues.input[trainingValuesIndex] + ',';
        }
        console.log(out);

        out = 'Test expected output: ';
        for (var trainingValuesIndex = 0;trainingValuesIndex < trainingValues.input.length;trainingValuesIndex ++) {
          out += trainingValues.expectedOutput[trainingValuesIndex] + ',';
        }
        console.log(out);

        for (var nodeIndex = 0;nodeIndex < inputLayer.getNodeCount();nodeIndex ++) {
          inputLayer.setNodeOutput(nodeIndex, trainingValues.input[nodeIndex]);
        }
        this.forward();

        out = 'Outputs: '
        for (var nodeIndex = 0;nodeIndex < outputLayer.getNodeCount();nodeIndex ++) {
          out += outputLayer.getNodeOutput(nodeIndex) + ',';
        }
        console.log(out);

        this.evaluateError(trainingValues);
      }
    },
    forward: function() {
      for (var index = 1;index < this.layers.length;index ++) {
        this.layers[index].forward();
      }
    },
    displayToConsole: function() {
      for (var index = 1;index < this.layers.length;index ++) {
        this.layers[index].displayToConsole();
      }
      console.log('');
    }
  }
};

// Initialize our very small network
var theNetwork = new network();
theNetwork.initialize(4, 3);

// Train it
theNetwork.train(20000000);
theNetwork.testAgainstTrainingData();

