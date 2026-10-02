# Reward servers

The VVR reward and the OCR and PickScore rewards run inside the training process. GenEval, GenEval2, and
UnifiedReward run as HTTP servers that the trainer calls. Only the runs whose reward includes them need the
servers:

| Reward | Runs | Server | Trainer environment variable |
|---|---|---|---|
| GenEval | `five_reward`, `five_reward_vvr_easy` | `geneval_server.py` (Mask2Former + CLIP), port 18085 | `GENEVAL_SERVER_URLS` |
| GenEval2 | `geneval2`, `geneval2_vvr_easy`, `geneval2_vvr_matched`, `five_reward`, `five_reward_vvr_easy` | `geneval2_server.py` (Qwen3-VL-2B Soft-TIFA, geometric mean), port 18185 | `GENEVAL2_SERVER_URLS` |
| UnifiedReward | `five_reward`, `five_reward_vvr_easy` | vLLM serving `CodeGoat24/UnifiedReward-2.0-qwen3vl-8b`, port 8080 | `UNIFIEDREWARD_BASE_URLS` |

Each variable takes a comma-separated list of URLs. Each rank starts from a different URL in the list.
Without the variable, the trainer uses the default local port.

```bash
python reward_servers/geneval_server.py --port 18085
python reward_servers/geneval2_server.py --port 18185   # needs a transformers version with Qwen3-VL

vllm serve CodeGoat24/UnifiedReward-2.0-qwen3vl-8b \
  --host 0.0.0.0 --port 8080 --trust-remote-code \
  --served-model-name UnifiedReward --max-model-len 8192 \
  --limit-mm-per-prompt.image 4
```

The OCR reward uses PaddleOCR (`pip install paddleocr`). The PickScore reward loads
`yuvalkirstain/PickScore_v1`; set `PICKSCORE_MODEL_PATH` to use a local copy.
