
# ViT 

The Vision Transformer (ViT) model was introduced in 2021 in a conference research paper titled “An Image is Worth 16*16 Words:	
Transformers for Image Recognition at Scale,” published at ICLR 20211. The ViT model is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder2.


![The-Vision-Transformer-architecture-a-the-main-architecture-of-the-model-b-the](https://github.com/prashantgautam96/ViT_implementation/assets/92217129/452f87a7-03d8-4abb-b5db-9864e641a607)






## Implementation

**1.** Define the input image size, the number of patches, the patch size, the hidden dimension, the number of heads, the number of blocks, and the output dimension. These are the hyperparameters that will determine the shape and size of the model.

**2.**: Define a function to split an image into patches of equal size. Each patch will be flattened into a vector and concatenated with a learnable embedding vector. The result will be a matrix of shape (n_patches, patch_size * patch_size + hidden_d).

**3.**: Define a function to add positional embeddings to the patch embeddings. This is to provide some information about the spatial location of each patch in the image. The positional embeddings are also learnable vectors of the same shape as the patch embeddings. The result will be a matrix of shape (n_patches, patch_size * patch_size + hidden_d).

**4.**: Define a class for the Multi-Head Self-Attention (MSA) block. This is the core component of the ViT model that allows each patch to attend to every other patch in the image. The MSA block consists of four steps:

- Split the input matrix into n_heads sub-matrices along the hidden dimension axis. This is to allow parallel computation of multiple attention heads.
- Compute three matrices called queries, keys, and values for each sub-matrix by multiplying them with learnable weight matrices. These are used to measure the similarity and importance of each patch in relation to others.
- Compute the attention scores for each pair of patches by taking the dot product of queries and keys, and applying a softmax function. This is to normalize the scores and make them sum up to one for each query.
- Compute the output matrix for each sub-matrix by taking the weighted sum of values according to the attention scores. This is to aggregate the information from all patches based on their relevance.
- Concatenate the output matrices from all sub-matrices along the hidden dimension axis, and apply a linear transformation with another learnable weight matrix. This is to restore the original shape and dimension of the input matrix.

**5.**: Define a class for the Multi-Layer Perceptron (MLP) block. This is another component of the ViT model that applies two linear transformations with a non-linear activation function (GELU) in between. The MLP block also has a residual connection that adds the input matrix to the output matrix. This is to help with gradient flow and avoid vanishing gradients.

**6.**: Define a class for the Transformer Encoder block. This is a combination of an MSA block and an MLP block, with layer normalization applied before and after each block. The Transformer Encoder block also has residual connections that add the input matrix to the output matrix of each block. This is to further improve gradient flow and model performance.

**7.**: Define a class for the ViT model. This is a sequence of patch embedding, positional embedding, and n_blocks Transformer Encoder blocks, followed by a classification head that consists of layer normalization, global average pooling, and a linear layer. The ViT model takes an image as input and outputs a vector of logits for each class.




## Run Locally

Clone the project

```bash
  git clone https://github.com/prashantgautam96/ViT_implementation
```

Go to the project directory

```bash
  cd ViT_implementation
```

Install dependencies

```bash
  pip install numpy
  pip install tqdm
  pip install torch
  pip install torchvision
```

Train & Test the ViT Model

```bash
  Click on the play button of each module 
```

## Real World Implementations

The ViT model has shown promising results in various computer vision tasks, such as image classification, object detection, and semantic image segmentation. Here are some potential future ideas for ViT transformation:

1. **Video Classification**: ViT can be extended to video classification tasks by treating each frame of a video as an image and applying the ViT model to each frame.

2. **Image Generation**: ViT can be used to generate images by training a generative model on the learned representations of the ViT model.

3. **Transfer Learning**: ViT can be used as a pre-trained model for transfer learning on other computer vision tasks.

4. **Multimodal Learning**: ViT can be combined with other models, such as language models, to perform multimodal learning tasks.
    
