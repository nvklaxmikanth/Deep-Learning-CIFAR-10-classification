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

### Dependencies

This project uses python. Hence python version 3.7.xx or higher should be installed. We recommend installing with Anaconda and adding it to the path of Windows. Git is also recommended to easily download files from GitHub. Alternatively you can also download the folder from GitHub directly. The links for both are given below
* [Python](https://www.python.org/)
* [Git](https://git-scm.com/)

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

Residual blocks can be defined as a function 
xl+1 = xl + F (xl, Wl) (1).
where, xl+1 and xl are input and output of the l-th layer of the ResNet network, F is the residual function and W are the block parameters. The ResNet consists of N residual layers, where each layer consists of one or more residual blocks. The input to the network is a tensor Xinput with shape < Hi, Wi, Ci >, where Hi and Wi are the spatial dimensions and Ci is the number of channels. 

Our goal is to maximize test accuracy for a given resource constraint. The model parameters we are allowed to change are - 
1. Number of residual layers, 
2. Number of blocks in residual layer i, 
3. Number of channels in residual layer i 
4. Convolutional kernel size in residual layer i, 
5. Skip connection kernel size in residual layer i, and 
6. Average pool kernel size.

In this project, we adopt the random search technique for hyperparameter tuning. The main goal of our learning algorithm is to find a function fθ that minimizes the pre-defined loss L(x; fθ ) over a dataset Xtrain. The learning algorithm computes f through the optimization of a set of parameters θ. The random search technique involves defining a search space as a bounded domain for the model hyperparameters and randomly sample points in that domain.

![Random Search](/images/num_block_inplane_effects.png)

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

<p align="right">(<a href="#top">back to top</a>)</p>

## 📌 Key Architectural Components

- **Input Layer** → Accepts an image of size \((3 \times 32 \times 32)\)
- **Initial Convolution** → 3×3 Conv with **16 filters**, BatchNorm, ReLU
- **Residual Layers**  
  - **Layer 1:** Multiple **BasicBlocks**, 16 filters  
  - **Layer 2:** 32 filters, **stride = 2** (reduces spatial dimensions)  
  - **Layer 3:** 64 filters, **stride = 2**  
- **BasicBlock Design** → 2×(3×3 Conv + BatchNorm + ReLU) with **skip connections**
- **Shortcut Connections** → Uses **1×1 Conv** when dimensions mismatch
- **Global Average Pooling** → Reduces feature maps to **single vectors**
- **Fully Connected Layer** → Maps features to class scores
- **Softmax Output** → Predicts probabilities for **10 classes**

## 📊 Network Architecture & Hyperparameter Optimization

### 🔬 Depth of the Network
- More layers improve hierarchical feature extraction but can cause **vanishing gradients**  
- We optimize depth to **balance performance & efficiency**  

### 📏 Width of Residual Blocks
- More channels → **Better feature representation**  
- Wider models are often **more efficient** than deeper models  

### 🖼️ Data Augmentation Techniques
To enhance **generalization** and prevent **overfitting**, we apply:
**Random Cropping** → Extracts **32×32** patches 📦  
**Random Rotation** → Applies **−5° to +5°** rotations 🔄  
**Random Horizontal Flip** → 50% chance of flipping ↔
