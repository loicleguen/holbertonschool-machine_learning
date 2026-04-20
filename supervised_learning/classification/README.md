<div align="center"><img src="https://github.com/ksyv/holbertonschool-web_front_end/blob/main/baniere_holberton.png"></div>

# Background Context

## Table of Contents :

  - [0. Neuron](#subparagraph0)
  - [1. Privatize Neuron](#subparagraph1)
  - [2. Neuron Forward Propagation](#subparagraph2)
  - [3. Neuron Cost](#subparagraph3)
  - [4. Evaluate Neuron](#subparagraph4)
  - [5. Neuron Gradient Descent](#subparagraph5)
  - [6. Train Neuron](#subparagraph6)
  - [7. Upgrade Train Neuron](#subparagraph7)
  - [8. NeuralNetwork](#subparagraph8)
  - [9. Privatize NeuralNetwork](#subparagraph9)
  - [10. NeuralNetwork Forward Propagation](#subparagraph10)
  - [11. NeuralNetwork Cost](#subparagraph11)
  - [12. Evaluate NeuralNetwork](#subparagraph12)
  - [13. NeuralNetwork Gradient Descent](#subparagraph13)
  - [14. Train NeuralNetwork](#subparagraph14)
  - [15. Upgrade Train NeuralNetwork](#subparagraph15)

## Resources
### Read or watch:
* [Supervised vs. Unsupervised Machine Learning](/rltoken/GODndrOYXluOorsgye-EQQ)
* [How would you explain neural networks to someone who knows very little about AI or neurology?](/rltoken/4GZq0g5rkWwbpdR0qhfK4Q)
* [Using Neural Nets to Recognize Handwritten Digits](/rltoken/74uz43zF7FFBcrTWRvxWPg)
* [Forward propagation](/rltoken/yG3ZwTBl-xd174Sf8V3P8w)
* [Understanding Activation Functions in Neural Networks](/rltoken/osiJrbwSVFWA_lMDrBkS3Q)
* [Loss function](/rltoken/2H5h6wuJjozYAZeB9nO2cg)
* [Gradient descent](/rltoken/shkmz4JXewQwBXO6CRIWMw)
* [Calculus on Computational Graphs: Backpropagation](/rltoken/NKqjU_4Gv9BVjEZ9NSY6Zg)
* [Backpropagation calculus](/rltoken/rkan8x6RV5yf-I6e4HmysQ)
* [What is a Neural Network?](/rltoken/d1n9LNe-Hir3SwZEtfkJ2A)
* [Supervised Learning with a Neural Network](/rltoken/HnbwRV8aZ5QFCpA4W0Ltxw)
* [Binary Classification](/rltoken/TzET6n9xkveAkX7vjAlV0g)
* [Logistic Regression](/rltoken/kEePCtIIcEuxE_Z0v91MPg)
* [Logistic Regression Cost Function](/rltoken/5IocHKVJNe8s1Y_Kkwo9yQ)
* [Gradient Descent](/rltoken/w1GYwJCiQ9fPeaLhwRCBFg)
* [Computation Graph](/rltoken/P-rKFRECb5zVzjbw9-shTA)
* [Logistic Regression Gradient Descent](/rltoken/2yxJUi6IxCcI9o_adS3Nfg)
* [Vectorization](/rltoken/vzgieyb-79Bai6t4erJWLA)
* [Vectorizing Logistic Regression](/rltoken/7LLxwZYFO91mval1rr1LPg)
* [Vectorizing Logistic Regression's Gradient Computation](/rltoken/4cTZ3wDiQMjHZEXKjjRUzQ)
* [A Note on Python/Numpy Vectors](/rltoken/tO_xks02h7nULzGW2ot0nA)
* [Neural Network Representations](/rltoken/It_etMoyIZTpGoWPuto4qQ)
* [Computing Neural Network Output](/rltoken/byB7ooxeCvKm-EWxQXPnGw)
* [Vectorizing Across Multiple Examples](/rltoken/8DlG08kGM9G3az4rsnS7lQ)
* [Gradient Descent For Neural Networks](/rltoken/MVeTO5Svp67ch_2oU02f_g)
* [Random Initialization](/rltoken/w9-MIiVApd1Vg2Yn-u-CEw)
* [Deep L-Layer Neural Network](/rltoken/AYtLg-EW2J9XippK5yktpQ)
* [Train/Dev/Test Sets](/rltoken/4vi6B1zg6YjOodBhxsrzBQ)
* [Random Initialization For Neural Networks : A Thing Of The Past (Includes He et al. initialization)](/rltoken/G8jTAavqDb6I8mKavDGyHA)
* [Initialization of deep networks](/rltoken/wqFe0AJj0En68GJucO-kRw)
* [Multiclass classification](/rltoken/PjU_ZIRPx2Lnbr_QInCnfg)
* [Derivation: Derivatives for Common Neural Network Activation Functions](/rltoken/eLNzx6nlB5WUbb-PONSUJw)
* [What is One Hot Encoding? Why And When do you have to use it?](/rltoken/tQLvNcnoQgJumwlvSTfb3A)
* [Softmax function](/rltoken/3gy1sYS5jcKdhf_pTFpfCA)
* [What is the intuition behind SoftMax function?](/rltoken/0EiPN0puNedYSqHWNaLX2w)
* [Cross entropy](/rltoken/murnOC7lOP-itLtj9WJ0Ig)
* [Loss Functions: Cross-Entropy](/rltoken/t_3VAN3eIvRvD9vZcqKyyA)
* [Softmax Regression](/rltoken/kHH8I-9wkJL87m6RO8z50Q)
* [Training Softmax Classifier](/rltoken/lYXxaiL-AaXdl9gr6tENyg)
* [numpy.zeros](/rltoken/RUsQZw6JlCAQ2RA73f2M1Q)
* [numpy.random.randn](/rltoken/dXaNmIbxCuT7orcIytk2Qg)
* [numpy.exp](/rltoken/-I27hkzghlX3iKRvAJCtug)
* [numpy.log](/rltoken/qhLHcLyVRvE5htuTyTMDMA)
* [numpy.sqrt](/rltoken/qHCUdqJsayukUlWnRoT9AQ)
* [numpy.where](/rltoken/0I6i19i9-aKhmo-x-Y4Inw)
* [numpy.max](/rltoken/sn5Z--laJuXjWdEFzgk1FQ)
* [numpy.sum](/rltoken/e-RPkC6dimMbZFyJ_AA0ig)
* [numpy.argmax](/rltoken/K91hSQ249oktCzOJidvAyg)
* [What is Pickle in python?](/rltoken/TboK-Jf1rvySku07S65BfA)
* [pickle](/rltoken/M89rqHR0iyO-capObEkPxg)
* [pickle.dump](/rltoken/Rhq6-8HB3HLat-_4U9j3ug)
* [pickle.load](/rltoken/OODfkwCN6sCfERErwlS64g)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:
* What is a model?
* What is supervised learning?
* What is a prediction?
* What is a node?
* What is a weight?
* What is a bias?
* What are activation functions?Sigmoid?Tanh?Relu?Softmax?
* Sigmoid?
* Tanh?
* Relu?
* Softmax?
* What is a layer?
* What is a hidden layer?
* What is Logistic Regression?
* What is a loss function?
* What is a cost function?
* What is forward propagation?
* What is Gradient Descent?
* What is back propagation?
* What is a Computation Graph?
* How to initialize weights/biases
* The importance of vectorization
* How to split up your data
* What is multiclass classification?
* What is a one-hot vector?
* How to encode/decode one-hot vectors
* What is the softmax function and when do you use it?
* What is cross-entropy loss?
* What is pickling in Python?

## Requirements
### General
* Allowed editors:vi,vim,emacs
* All your files will be interpreted/compiled on Ubuntu 20.04 LTS usingpython3(version 3.9)
* Your files will be executed withnumpy(version  1.25.2)
* All your files should end with a new line
* The first line of all your files should be exactly#!/usr/bin/env python3
* AREADME.mdfile, at the root of the folder of the project, is mandatory
* Your code should use thepycodestylestyle (version 2.11.1)
* All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
* All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
* All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)'andpython3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
* Unless otherwise noted, you are not allowed to import any module exceptimport numpy as np
* Unless otherwise noted, you are not allowed to use any loops (for,while, etc.)
* All your files must be executable
* The length of your files will be tested usingwc

## Task
### 0. Neuron <a name='subparagraph0'></a>

---

### 1. Privatize Neuron <a name='subparagraph1'></a>

---

### 2. Neuron Forward Propagation <a name='subparagraph2'></a>

---

### 3. Neuron Cost <a name='subparagraph3'></a>

---

### 4. Evaluate Neuron <a name='subparagraph4'></a>

---

### 5. Neuron Gradient Descent <a name='subparagraph5'></a>

---

### 6. Train Neuron <a name='subparagraph6'></a>

---

### 7. Upgrade Train Neuron <a name='subparagraph7'></a>

---

### 8. NeuralNetwork <a name='subparagraph8'></a>

---

### 9. Privatize NeuralNetwork <a name='subparagraph9'></a>

---

### 10. NeuralNetwork Forward Propagation <a name='subparagraph10'></a>

---

### 11. NeuralNetwork Cost <a name='subparagraph11'></a>

---

### 12. Evaluate NeuralNetwork <a name='subparagraph12'></a>

---

### 13. NeuralNetwork Gradient Descent <a name='subparagraph13'></a>

---

### 14. Train NeuralNetwork <a name='subparagraph14'></a>

---

### 15. Upgrade Train NeuralNetwork <a name='subparagraph15'></a>

---


## Authors
loicleguen - [GitHub Profile](https://github.com/loicleguen)
