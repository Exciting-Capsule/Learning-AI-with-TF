# TensorFlow.js Important Key Concepts
This is a very very brief summary of Concepts used in TensorFlow.js and suppose to be as surface and relatable as possible. 
Please correct me/this if there's any inaccuracies in the understanding of the topic.

1. Tensors
  - Tensors is the central unit of data.
  - It contains a set of numerical value shaped into an array.
  - It can be array of one to more dimensions as follow
  	Eg: 1d - [1,2,3,4]
  		2d - [ [1,2],[3,4] ]
  		3d - [  [ [1],[2] ],[ [3],[4] ]  ]

2. Variables
  - Variables are just like Tensors except their values are mutable.

3. Operations (Ops)
  - Operations literally mean operation done on your data(tensors/variable), it will manipulate your data.
  - TensorFlow.js already provides tons of ops for machine learning.
  - Operations also contains basic arithmetic such as
    Eg: add, sub, mul, square

4. Models and Layers
  - Model is like a function. We provide input to a model and it will return us some output.
  - Layers are consist of **Input Layer**, **Hidden Layer** and **Output Layer**.
  	- Input layer is like argument to a function
  	- Hidden layer is like whatever operations that's done on the argument of the function (add, sub, etc.)
  	- Output layer is like the return of a function
  - There can only be one Input layer and Output layer, but Hidden layer can be as many as possible depending on the neural network.


P.S. Please be reminded again this is just a summary and brief ideas of whatever used in TensorFlow.js. For more information, you can always read it from this [link](https://js.tensorflow.org/tutorials/core-concepts.html)