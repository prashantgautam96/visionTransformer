
# ViT 

The Vision Transformer (ViT) model was introduced in 2021 in a conference research paper titled “An Image is Worth 16*16 Words:	
Transformers for Image Recognition at Scale,” published at ICLR 20211. The ViT model is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder2.




![Logo](https://www.researchgate.net/publication/348947034/figure/fig2/AS:986572736446471@1612228678819/The-Vision-Transformer-architecture-a-the-main-architecture-of-the-model-b-the.png)

## Implementation

- An image is split into fixed-size patches.
- Each patch is linearly embedded.
- Position embeddings are added to the resulting sequence of vectors.
- The sequence of vectors is fed to a standard Transformer encoder.
In order to perform classification, an extra learnable “classification token” is added to the sequence. The resulting sequence is then fed to a standard Transformer decoder.


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
    
