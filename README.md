# Deep-Learning-CIFAR-10-classification

[Geetha Krishna Guruju](gg3039@nyu.edu), [NVK Laxmi Kanth](vn2263@nyu.edu), [Jeevana Bhumireddy](jb8855@nyu.edu),

New York University


## Introduction

![Architechture](/images/architechture.png)

Deep learning has revolutionized image classification, with Residual Networks (ResNets) emerging as a cornerstone in achieving high accuracy on complex datasets. By incorporating skip connections, ResNets effectively mitigate the vanishing gradient problem, allowing for deeper architectures that extract rich hierarchical features. However, as model depth increases, so does the computational cost, making it essential to strike a balance between accuracy and efficiency.

In this project, we focus on designing a modified ResNet architecture that optimizes classification accuracy on the CIFAR-10 dataset while ensuring that the total number of parameters remains under 5 million. This constraint presents a unique challenge, requiring careful architectural modifications to maximize performance without excessive complexity. We explore key hyperparameters, including the number of residual layers, block configurations, channel dimensions, kernel sizes, and pooling strategies, to develop an efficient yet powerful model.

The proposed architecture leverages residual learning to enhance gradient flow, optimize feature extraction, and improve generalization. By systematically analyzing the impact of different design choices, we develop a model that achieves competitive accuracy while maintaining a lightweight structure. Our findings contribute to the broader understanding of efficient ResNet architectures and their applicability in resource-constrained environments.

Refer the `README.md` to get started.

<p align="right">(<a href="#top">back to top</a>)</p>

### Implementation

This project is built with the below given major frameworks and libraries. The code is primarily based on python. Some plugins (Weights and Biasis and Matplotlib) are added for easy visualization of the work (and is not needed for training).

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Weights & Biases](https://wandb.ai/site)
* [Matplotlib](https://matplotlib.org/)
* [Cifar-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To reproduce the results and to download the model follow the procedure in this section. 


### Installation

_How to reproduce the result and or clone the repository_

1. Clone the repo
   ```sh
   git clone https://github.com/geethaguruju/Deep-Learning-CIFAR-10-classification.git
   ```
2. Install requirements
   ```sh
   pip3 install requirements.txt
   ```
3. Run train script `train.py` to recreate similar model
   ```sh
   python train.py
   ```
4. To generate the predictions for the test data
   ```sh
   python infer.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


## Methodology

Residual blocks can be defined as a function:

```math
x_{l+1} = x_l + F(x_l, W_l)
```

where, \(x_{l+1}\) and \(x_l\) are the input and output of the \(l\)-th layer, \(F\) is the residual function, and \(W\) are the block parameters. ResNet consists of \(N\) residual layers, each containing one or more residual blocks. The input to the network is a tensor \(X_{input}\) with shape \((H_i, W_i, C_i)\), where \(H_i\) and \(W_i\) are spatial dimensions, and \(C_i\) is the number of channels.

### Key Optimized Parameters
- Number of residual layers
- Number of blocks in each residual layer
- Number of channels per residual layer
- Convolutional kernel size in each residual layer
- Skip connection kernel size
- Average pooling kernel size

### Hyperparameter Tuning
We use **random search** to optimize hyperparameters, defining a bounded search space and selecting random configurations to improve accuracy and minimize loss.

# Results

The model exhibited a steady improvement in both training and test accuracy throughout training. Below are key observations and a summary of accuracy progression:


### Key Observations

- **Training Accuracy** (solid blue line) consistently increased, approaching **100%** by the end of training.
- **Test Accuracy** (dashed orange line) showed fluctuations but remained above **90%** after 50 epochs, stabilizing at **94.60%** after 200 epochs.
- The trends observed in the accuracy table align with the plotted accuracy curve, confirming consistency in recorded values.

## Training Progress and Accuracy Improvements

| Epoch | Train Accuracy | Test Accuracy |
|-------|--------------|--------------|
| 0     | 27.91%       | 50.60%       |
| 50    | 92.25%       | 90.59%       |
| 100   | 95.65%       | 91.98%       |
| 200   | 99.92%       | **94.60%**   |

<p align="right">(<a href="#top">back to top</a>)</p>
