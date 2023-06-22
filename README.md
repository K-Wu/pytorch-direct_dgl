# PyTorch-Direct
## Introduction
PyTorch-Direct adds a zero-copy access capability for GPU on top of the existing PyTorch DNN framework. Allowing the zero-copy access capabily for GPU significantly increases the data transfer efficiency over PCIe when the targeted data is scattered in the host memory. This is especially useful when the input data cannot be fit into the GPU memory ahead of the training time, and data pieces need to be transferred during the training time. With PyTorch-Direct, using the zero-copy access capability can be done by declaring a "Unified Tensor" on top of the existing CPU tensor. The current implementation of PyTorch-Direct is based on the nightly version of PyTorch-1.8.0.

## Installation

### Pytorch
Since we modify the source code of PyTorch, our implementation cannot be installed through well-known tools like `pip`. To compile and install the modified version of our code, please follow [this](https://github.com/K-Wu/pytorch-direct/tree/e2d0a3366145d0df4577797a5b2117c69271009c#from-source).

### DGL
dgl 0.6.1
`conda install -c dglteam "dgl-cuda11.3<0.7"`

## Use case
In the original PyTorch, the scattered data in the host can be accessed by the GPU like the following example:
```
# input_tensor: A given input 2D tensor in CPU
# index: A tensor which has indices of targets
# output_tensor: An output tensor which should be located in GPU

output_tensor = input_tensor[index].to(device="cuda:0")
```

Now in PyTorch-Direct, the code can be transformed into as follows:
```
# input_tensor: A given input 2D tensor in CPU
# index: A tensor which has indices of targets
# output_tensor: An output tensor which should be located in GPU

torch.cuda.set_device("cuda:0")
unified_tensor = input_tensor.to(device="unified")

output_tensor = unified_tensor[index]
```

The unified tensor does not physically copy any data, but only creates a mapping for the GPU. Therefore, in current implementation, if the original CPU tensor disappears, the unified tensor which created later cannot be accessed.

For such reason, the following coding practice should be avoided for now:
```
output_tensor = torch.randn([100,100], device="cpu").to(device="unified")
```

A temporary tensor created by the `randn` function will disappear as it is not assigned to any. The unified tensor created by the following code has no physical data therefore. The code should be re-written as follows:
```
temp_tensor = torch.randn([100,100], device="cpu")
output_tensor = temp_tensor.to(device="unified")
```
In this case the temporary tensor is fixed to `temp_tensor` declaration so the unified tensor can be safely called on it.

## GNN Example
### Basics
For a more practical example, we perform GNN training with the well known Deep Graph Library (DGL). The example code is located in the dgl submodule of this repository. The exact location is `<current_path>/dgl/examples/pytorch/graphsage/train_sampling_pytorch_direct.py`.
To compare with the original PyTorch approach, the users can use the unmodified DGL implementation in `<current_path>/dgl/examples/pytorch/graphsage/train_sampling.py`. By default, the DGL example always try to load the whole data into the GPU memory. Therefore, to compare the host memory access performance, the user needs to add `--data-cpu` argument to the DGL example.

### DGL Installation
We do not modify the source of DGL so the users can either install DGL using `pip` or by compiling from the source code. Please follow the DGL readme file to install DGL.

### Using Multi-Processing Service (MPS)
To further increase the efficiency of PyTorch-Direct in GNN training, CUDA Multi-Processing Service (MPS) can be used. The purpose of MPS is to allocate a small amount of GPU resource for the zero-copy accesses while leaving the rest for the training process. The MPS can be used in our example GNN code by passing `--mps x,y` argument. Here, `x` is the GPU portion given for the zero-copy kernel and `y` is the GPU portion given for the trainig process. For the NVIDIA RTX 3090 GPU we used, we used `--mps 10,90` setting.
Using MPS requires running an external utility called `nvidia-cuda-mps-control`. This utiliy should be available as far as CUDA is installed. Running `nvidia-cuda-mps-control` does not require a root permission as the restriction is only applied to the users who are using it. In `<current_path>/dgl/examples/pytorch/graphsage/utils.py` file, we added some scripts which deal with running MPS. The functions declared in this file are used inside `<current_path>/dgl/examples/pytorch/graphsage/train_sampling_pytorch_direct.py`.

### Quick Evaluation
![Reddit](https://github.com/K-Wu/pytorch-direct_dgl/blob/master/docs/reddit.png)\
In this chart, we show a GraphSAGE training result for the reddit dataset. Since the reddit dataset is small and can be located either in the host memory or the GPU memory, we tested both cases. For the evaluation, we used AMD Threadripper 3960x CPU and NVIDIA RTX 3090 GPU. As we can observe, with a faster interconnect, the benefit of PyTorch-Direct is greater and it can nearly reach the all-in-GPU memory case.

## Citation
```
@article{min2021large,
  title={Large Graph Convolutional Network Training with GPU-Oriented Data Communication Architecture},
  author={Min, Seung Won and Wu, Kun and Huang, Sitao and Hidayeto{\u{g}}lu, Mert and Xiong, Jinjun and Ebrahimi, Eiman and Chen, Deming and Hwu, Wen-mei},
  journal={arXiv preprint arXiv:2103.03330},
  year={2021}
}
```
