
# ViT 

The Vision Transformer (ViT) model was introduced in 2021 in a conference research paper titled “An Image is Worth 16*16 Words:	
Transformers for Image Recognition at Scale,” published at ICLR 20211. The ViT model is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder2.




![Logo](https://learnopencv.com/wp-content/uploads/2023/02/image-9.png)


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
