# RETFound weights (protocol Branch A2)

Official repo: https://github.com/rmaphoh/RETFound (cloned to `external/RETFound`).

Pre-trained CFP weights are **gated** on Hugging Face:

1. Request access: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
2. Login: `huggingface-cli login` (or set `HF_TOKEN`)
3. Train: see `train_retfound.py` / `protocol_autonomous.sh` run `protocol_r19_retfound_excl_t1`

Local checkpoint override:

```bash
python train_retfound.py \
  --finetune /path/to/RETFound_mae_natureCFP.pth \
  ...
```
