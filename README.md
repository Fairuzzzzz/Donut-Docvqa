## Donut-DocVQA: Document Visual Question Answering with Donut Model

### Overview

This repository contains the implementation of a Document Visual Question Answering (DocVQA) system using the Donut model. The Donut model is a state-of-the-art vision-language model designed to understand and process documents visually. This project leverages the naver-clova-ix/donut-base model to answer questions based on the content of document images.

### Table of Contents
- Instalation
- Dataset
- Training the Model
- Evaluation
- Training
- Evaluation
- Model Deployment

### Instalattion
To get started, you need to install the required dependencies. You can do this by running the following commands:
```
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q datasets sentencepiece
pip install -q pytorch-lightning wandb
```
These commands will install the necessary libraries, including Hugging Face Transformers, Datasets, SentencePiece, PyTorch Lightning, and Weights and Biases (WandB) for logging.

### Dataset
### Dataset Structure
The dataset used in this project is `indra-inc/docvqa_en_train_valid_2400_gtparse`, which consists of document images paired with corresponding questions and answers. The dataset is split into two parts: train and valid. Each sample in the dataset contains the following fields:

    - question: The question asked about the content of the document.
    - docId: A unique identifier for the document.
    - answers: The correct answer(s) to the question.
    - data_split: Indicates whether the sample belongs to the training or validation split.
    - bounding_boxes: The bounding boxes that outline relevant regions in the document image.
    - word_list: A list of words recognized in the document image.
    - image_raw: The raw image of the document.
    - ground_truth: The ground truth annotations, often in JSON format, which include information needed to generate the correct answer.

### Training the Model

The training process involves fine-tuning the Donut model on the DocVQA dataset. We use PyTorch Lightning to handle the training loop. Below is the code to initialize the training:

```
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Configuration and model setup
config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)

# Custom dataset class
train_dataset = DonutDataset("indra-inc/docvqa_en_train_valid_2400_gtparse", max_length=128, split="train")
val_dataset = DonutDataset("indra-inc/docvqa_en_train_valid_2400_gtparse", max_length=128, split="valid")

# DataLoader setup
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# PyTorch Lightning module setup
model_module = DonutModelPLModule(config, processor, model)
wandb_logger = WandbLogger(project="Donut-DocVQA")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=10,
    val_check_interval=0.2,
    check_val_every_n_epoch=1,
    gradient_clip_val=1.0,
    precision=16,  # we'll use mixed precision
    num_sanity_val_steps=0,
    logger=wandb_logger,
)

# Start training
trainer.fit(model_module)

```

### Evaluation
After training, the model can be evaluated on the validation set. Evaluation involves generating answers for each question in the validation set and comparing them with the ground truth.

### Model Deployment
You can deploy the trained model to Hugging Face Hub for easy sharing and further use:
```
repo_name = "your-username/donut-docvqa" # change with your repository

model_module.processor.push_to_hub(repo_name)
model_module.model.push_to_hub(repo_name)

```
