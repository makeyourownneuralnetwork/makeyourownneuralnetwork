# makeyourownneuralnetwork
This is the code from the [Make Your Own Neural Network book](https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G) converted into a module. It works with Python 2.7+.

Place the module in the working directory and import it with:

```python
import neuralNetwork as nn
```

Build a nn with:

```python
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = nn.neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
```
