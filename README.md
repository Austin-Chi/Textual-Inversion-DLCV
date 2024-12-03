# Textual Inversion on a Corgi and the style of David Revoy

There are two textual inversion embeddings, one for the corgi and one for the style.

## Usage

### Training
```bash
bash get_dataset.sh
python3 stable-diffusion/textual_inversion.py
```
Go to `stable-diffusion/scripts/textual_inversion.py `(for object personalization), follow the args.
Go to `stable-diffusion/scripts/textual_inversion_2.py `(for style personalization), follow the args.

### Inferencing
Pretrained model:
```bash
bash hw2_download_ckpt.sh
```

Inference:
```bash
bash hw2_3.sh <path to the json file containing the testing prompt> <path to your output folder> <path to the pretrained model weight>
```

## Reference
[textual inversion](https://github.com/rinongal/textual_inversion/blob/main/main.py)

