# AIMv2 Speech Model

PyTorch implementation of AIMv2 (Autoregressive Image Models) adapted for speech domain, following the architecture from ["Multimodal Autoregressive Pre-training of Large Vision Encoders"](https://arxiv.org/abs/2411.14402).

## Key Features

- **Unified Multimodal Decoder**: Single decoder processes both speech and text concurrently
- **Prefix Attention Mask**: Enables bidirectional attention during inference without additional tuning
- **Modern Architecture**: SwiGLU FFN and RMSNorm throughout the model
- **Autoregressive Generation**: Causal attention for next-token prediction in both modalities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd aimv2

# Install dependencies
pip install torch transformers datasets torchaudio librosa
pip install tensorboard accelerate
```

## Usage

### Training

```bash
# Single GPU training
python train.py --config_path configs/aimv2_speech_config.json

# Multi-GPU training with DeepSpeed
deepspeed --num_gpus=8 train.py \
    --config_path configs/aimv2_speech_config.json \
    --deepspeed configs/deepspeed_config.json

# Resume from checkpoint
python train.py \
    --config_path configs/aimv2_speech_config.json \
    --resume_from_checkpoint outputs/aimv2_speech/checkpoint-5000
```

### Model Architecture

```python
from models import AIMv2Config, AIMv2Model

# Create model from config
config = AIMv2Config(
    encoder_embed_dim=1024,
    encoder_depth=24,
    encoder_num_heads=16,
    decoder_dim=1024,
    decoder_depth=24,
    decoder_num_heads=16,
    n_mels=128,
    patch_size=(16, 16),
    vocab_size=50257,
)

model = AIMv2Model(config)
```

### Inference

```python
import torch
from models import AIMv2Model

# Load pretrained model
model = AIMv2Model.from_pretrained("path/to/checkpoint")
model.eval()

# Prepare inputs
mel_spectrogram = torch.randn(1, 1, 128, 1000)  # [B, C, n_mels, time]
input_ids = tokenizer("transcribe this audio", return_tensors="pt").input_ids

# Forward pass
with torch.no_grad():
    outputs = model(
        mel_spectrogram=mel_spectrogram,
        input_ids=input_ids,
    )
    
# Access outputs
speech_logits = outputs.speech_logits  # Predicted speech patches
text_logits = outputs.text_logits      # Predicted text tokens
```

## Configuration

The main configuration file `configs/aimv2_speech_config.json` contains:

### Model Parameters
- `encoder_*`: Speech encoder settings (depth, heads, dimensions)
- `decoder_*`: Unified decoder settings
- `patch_size`: Size of speech patches (default: 16x16)
- `vocab_size`: Text vocabulary size (GPT-2: 50257)

### Training Parameters
- `batch_size`: Training batch size per GPU
- `learning_rate`: Peak learning rate (default: 1e-4)
- `warmup_steps`: Linear warmup steps
- `gradient_accumulation_steps`: Gradient accumulation for larger effective batch size

### Data Parameters
- `dataset_path`: Path to LAION-Audio dataset
- `sample_rate`: Audio sample rate (16kHz)
- `n_mels`: Number of mel-spectrogram bins (128)
- `max_audio_length`: Maximum audio duration in seconds

## Model Details

### Speech Encoder
- Processes mel-spectrograms with patch embedding
- Uses prefix attention mask during training:
  - Random prefix length M ~ U{1, 2, ..., I-1}
  - Bidirectional attention for prefix patches
  - Causal attention for non-prefix patches
- SwiGLU FFN and RMSNorm in all layers

### Unified Decoder
- Single transformer decoder for both modalities
- Causal attention mask for autoregressive generation
- Separate output heads:
  - Speech head: Predicts next speech patch (MSE loss)
  - Text head: Predicts next text token (Cross-entropy loss)

### Training Objective
- Combined loss: L = L_text + α * L_speech
- Speech loss computed only for non-prefix patches
- Both modalities trained concurrently

## Dataset

The model is designed to train on LAION-Audio-300M dataset. The dataset should be in Arrow format with the following structure:
```
dataset_path/
├── train/
├── validation/
└── test/
```

## Citation

If you use this code, please cite:
```bibtex
@article{aimv2,
  title={Multimodal Autoregressive Pre-training of Large Vision Encoders},
  author={...},
  journal={arXiv preprint arXiv:2411.14402},
  year={2024}
}
```