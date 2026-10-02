"""Rewards for RLVVR training.

`vvr_score` is the VVR training reward. The other rewards are the existing objectives that the paper mixes
with VVR; they are copied from the Flow-GRPO-based training code used for the paper. `multi_score` builds the
reward that the trainer calls.
"""

import asyncio
import base64
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

_REWARD_MISSING_VALUE = -10.0

def _coerce_numeric_vector(values, expected_len, key, context, *, allow_scalar_broadcast=False):
    if torch.is_tensor(values):
        values = values.detach().float().cpu().numpy()
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: metric {key} cannot be converted to an array") from exc
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    if arr.shape == ():
        if expected_len == 1 or allow_scalar_broadcast:
            return [float(arr.item())] * expected_len
        raise ValueError(
            f"{context}: metric {key} is scalar for batch length {expected_len}; "
            "reward metrics must be one value per sample"
        )
    flat = arr.reshape(-1)
    if len(flat) == expected_len:
        return [float(value) for value in flat]
    if len(flat) == 1 and allow_scalar_broadcast:
        return [float(flat[0])] * expected_len
    raise ValueError(
        f"{context}: metric {key} has length {len(flat)}, expected {expected_len}"
    )


def _is_numeric_singleton(values):
    if torch.is_tensor(values):
        values = values.detach().float().cpu().numpy()
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError):
        return False
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return False
    if arr.shape == ():
        return True
    return len(arr.reshape(-1)) == 1


def _coerce_reward_scores(scores, expected_len, context):
    values = _coerce_numeric_vector(
        scores,
        expected_len,
        "scores",
        context,
        allow_scalar_broadcast=False,
    )
    if values is None:
        raise ValueError(f"{context}: scores are non-numeric")
    return values


def _slice_batch_images(images, indices):
    if isinstance(images, torch.Tensor):
        return images[indices]
    return np.asarray(images)[indices]


def _fill_subset_details(details, sub_details, subset_indices, total_len):
    for key, values in (sub_details or {}).items():
        details.setdefault(key, [-10.0] * total_len)
        for local_idx, original_idx in enumerate(subset_indices):
            details[key][original_idx] = values[local_idx]


def pickscore_score(device):
    from rlvvr.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def ocr_score(device):
    from rlvvr.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = int(os.environ.get("GENEVAL_BATCH_SIZE", "64"))
    request_timeout = float(os.environ.get("GENEVAL_REQUEST_TIMEOUT", "300"))
    server_urls_env = os.environ.get("GENEVAL_SERVER_URLS", "").strip()
    if server_urls_env:
        server_urls = [u.strip().rstrip("/") for u in server_urls_env.split(",") if u.strip()]
        if not server_urls:
            raise ValueError("GENEVAL_SERVER_URLS was set but no valid URLs were parsed")
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        global_rank = int(
            os.environ.get(
                "RANK",
                os.environ.get("SLURM_PROCID", str(local_rank)),
            )
        )
        url = server_urls[global_rank % len(server_urls)]
        print(
            f"[GenEval] Rank {global_rank} local_rank {local_rank} using reward server {url}",
            flush=True,
        )
    else:
        url = "http://127.0.0.1:18085"
    retry_total = int(os.environ.get("GENEVAL_RETRY_TOTAL", "2"))
    retry_backoff = float(os.environ.get("GENEVAL_RETRY_BACKOFF", "1"))
    sess = requests.Session()
    retries = Retry(
        total=retry_total,
        backoff_factor=retry_backoff,
        status_forcelist=[500],
        allowed_methods=False,
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # Do not retry indefinitely: a rank-local hang here deadlocks
            # later distributed gathers.
            try:
                response = sess.post(url, data=data_bytes, timeout=request_timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"GenEval server {url} request failed after "
                    f"{retry_total} retries with timeout {request_timeout}s"
                ) from exc
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def geneval2_score(device):
    """Submit images to GenEval2 Soft-TIFA reward servers.

    Metadata must come from the GenEval2 jsonl rows and include `vqa_list`.
    The selected training score is controlled by GENEVAL2_METHOD:
    `soft_tifa_gm` by default, or `soft_tifa_am`.
    """
    import pickle
    import requests
    import threading
    from io import BytesIO
    from requests.adapters import HTTPAdapter, Retry

    batch_size = int(os.environ.get("GENEVAL2_BATCH_SIZE", "16"))
    request_timeout = float(os.environ.get("GENEVAL2_REQUEST_TIMEOUT", "300"))
    method = os.environ.get("GENEVAL2_METHOD", "soft_tifa_gm")
    server_urls_env = os.environ.get("GENEVAL2_SERVER_URLS", "").strip()
    if server_urls_env:
        server_urls = [u.strip().rstrip("/") for u in server_urls_env.split(",") if u.strip()]
        if not server_urls:
            raise ValueError("GENEVAL2_SERVER_URLS was set but no valid URLs were parsed")
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        global_rank = int(
            os.environ.get(
                "RANK",
                os.environ.get("SLURM_PROCID", str(local_rank)),
            )
        )
        url = server_urls[global_rank % len(server_urls)]
        print(
            f"[GenEval2] Rank {global_rank} local_rank {local_rank} using reward server {url}",
            flush=True,
        )
    else:
        server_urls = [os.environ.get("GENEVAL2_SERVER_URL", "http://127.0.0.1:18185").rstrip("/")]
        global_rank = 0

    retry_total = int(os.environ.get("GENEVAL2_RETRY_TOTAL", "2"))
    retry_backoff = float(os.environ.get("GENEVAL2_RETRY_BACKOFF", "1"))
    failover_rounds = int(os.environ.get("GENEVAL2_FAILOVER_ROUNDS", "2"))
    pool_maxsize = int(os.environ.get("GENEVAL2_HTTP_POOL_MAXSIZE", "64"))
    thread_state = threading.local()
    request_lock = threading.Lock()
    request_index = global_rank % len(server_urls)

    def _session():
        sess = getattr(thread_state, "session", None)
        if sess is None:
            retries = Retry(
                total=retry_total,
                backoff_factor=retry_backoff,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=False,
            )
            sess = requests.Session()
            adapter = HTTPAdapter(
                max_retries=retries,
                pool_connections=max(pool_maxsize, len(server_urls)),
                pool_maxsize=pool_maxsize,
            )
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            thread_state.session = sess
        return sess

    def _candidate_urls():
        nonlocal request_index
        with request_lock:
            start = request_index % len(server_urls)
            request_index += 1
        ordered = server_urls[start:] + server_urls[:start]
        return ordered

    def _fn(images, prompts, metadatas):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))

        all_scores = []
        details = defaultdict(list)
        for image_batch, metadata_batch in zip(images_batched, metadatas_batched):
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())
            payload = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batch),
                "method": method,
            }
            last_error = None
            response = None
            payload_bytes = pickle.dumps(payload)
            for round_idx in range(max(1, failover_rounds)):
                for url in _candidate_urls():
                    try:
                        response = _session().post(url, data=payload_bytes, timeout=request_timeout)
                        response.raise_for_status()
                        last_error = None
                        break
                    except requests.RequestException as exc:
                        last_error = exc
                if last_error is None:
                    break
                if retry_backoff > 0 and round_idx + 1 < max(1, failover_rounds):
                    import time

                    time.sleep(retry_backoff)
            if last_error is not None or response is None:
                raise RuntimeError(
                    "GenEval2 request failed on all configured reward servers after "
                    f"{max(1, failover_rounds)} failover rounds, per-server retry_total="
                    f"{retry_total}, timeout={request_timeout}s"
                ) from last_error
            response_data = pickle.loads(response.content)
            if "error" in response_data:
                raise RuntimeError(f"GenEval2 server failed: {response_data['error']}")
            all_scores.extend(response_data["scores"])
            for key in ("geneval2_am", "geneval2_gm", "geneval2_question_count", "geneval2_atom_count"):
                details[key].extend(response_data.get(key, []))
        return all_scores, dict(details)

    return _fn


def unifiedreward_score(device):
    """UnifiedReward-2.0 pointwise image-generation reward from an OpenAI-compatible (vLLM) server.

    Follows the UnifiedReward-2.0 pointwise contract: score a single image on Alignment, Coherence, and
    Style, each on a 1-5 scale. The scalar RL reward is the normalized mean of the three axes.
    """
    import asyncio
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AsyncOpenAI,
        InternalServerError,
    )
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=95)
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image/jpeg;base64,{encoded_image_text}"
        return base64_qwen

    def _build_pointwise_prompt(prompt):
        return (
            "You are presented with a generated image and its associated text caption. "
            "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
            "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms of content.\n"
            "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
            "Output your evaluation using the format below:\n\n"
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n\n"
            "Your task is provided as follows:\n"
            f"Text Caption: [{prompt}]"
        )

    def _extract_pointwise_scores(text_outputs):
        rewards = []
        details = {
            "unifiedreward_alignment": [],
            "unifiedreward_coherence": [],
            "unifiedreward_style": [],
            "unifiedreward_parse_success": [],
        }
        patterns = {
            "alignment": r"Alignment\s+Score\s*\(\s*1\s*-\s*5\s*\)\s*:\s*([0-5](?:\.\d+)?)",
            "coherence": r"Coherence\s+Score\s*\(\s*1\s*-\s*5\s*\)\s*:\s*([0-5](?:\.\d+)?)",
            "style": r"Style\s+Score\s*\(\s*1\s*-\s*5\s*\)\s*:\s*([0-5](?:\.\d+)?)",
        }

        for text in text_outputs:
            parsed = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, text or "", flags=re.IGNORECASE)
                try:
                    value = float(match.group(1)) if match else 0.0
                except (TypeError, ValueError):
                    value = 0.0
                parsed[key] = max(0.0, min(5.0, value))

            parse_success = all(parsed[key] > 0.0 for key in patterns)
            if parse_success:
                reward = float(np.mean([parsed["alignment"], parsed["coherence"], parsed["style"]]) / 5.0)
            else:
                reward = 0.0

            rewards.append(reward)
            details["unifiedreward_alignment"].append(parsed["alignment"] / 5.0)
            details["unifiedreward_coherence"].append(parsed["coherence"] / 5.0)
            details["unifiedreward_style"].append(parsed["style"] / 5.0)
            details["unifiedreward_parse_success"].append(1.0 if parse_success else 0.0)
        return rewards, details

    base_urls_env = os.environ.get("UNIFIEDREWARD_BASE_URLS", "").strip()
    if base_urls_env:
        base_urls = [u.strip().rstrip("/") for u in base_urls_env.split(",") if u.strip()]
        if not base_urls:
            raise ValueError("UNIFIEDREWARD_BASE_URLS was set but no valid URLs were parsed")
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        global_rank = int(
            os.environ.get(
                "RANK",
                os.environ.get("SLURM_PROCID", str(local_rank)),
            )
        )
        start_index = global_rank % len(base_urls)
        ordered_base_urls = (
            base_urls[start_index:] + base_urls[:start_index]
        )
        print(
            f"[UnifiedReward] Rank {global_rank} local_rank {local_rank} "
            f"using reward servers {ordered_base_urls}",
            flush=True,
        )
    else:
        ordered_base_urls = [
            os.environ.get(
                "UNIFIEDREWARD_BASE_URL", "http://127.0.0.1:8080/v1"
            ).rstrip("/")
        ]
    model_name = os.environ.get("UNIFIEDREWARD_MODEL", "UnifiedReward")
    timeout = float(os.environ.get("UNIFIEDREWARD_TIMEOUT", "180"))
    max_retries = int(os.environ.get("UNIFIEDREWARD_MAX_RETRIES", "3"))
    max_concurrent = int(os.environ.get("UNIFIEDREWARD_MAX_CONCURRENT", "16"))
    recovery_timeout = float(
        os.environ.get("UNIFIEDREWARD_RECOVERY_TIMEOUT", "600")
    )
    retry_backoff = float(
        os.environ.get("UNIFIEDREWARD_FAILOVER_BACKOFF", "2")
    )
    clients = [
        AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get("UNIFIEDREWARD_API_KEY", "flowgrpo"),
            timeout=timeout,
            max_retries=max_retries,
        )
        for base_url in ordered_base_urls
    ]
    retryable_errors = (APIConnectionError, APITimeoutError, InternalServerError)
        
    async def evaluate_image(prompt, image):
        question = _build_pointwise_prompt(prompt)
        images_base64 = pil_image_to_base64(image)
        request = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": int(os.environ.get("UNIFIEDREWARD_MAX_TOKENS", "512")),
        }
        loop = asyncio.get_running_loop()
        deadline = loop.time() + recovery_timeout
        failed_requests = 0
        last_error = None
        while True:
            for client in clients:
                try:
                    response = await client.chat.completions.create(**request)
                    return response.choices[0].message.content, failed_requests
                except retryable_errors as exc:
                    failed_requests += 1
                    last_error = exc
            if loop.time() >= deadline:
                raise RuntimeError(
                    "All UnifiedReward replicas remained unavailable for "
                    f"{recovery_timeout:.0f}s"
                ) from last_error
            await asyncio.sleep(min(retry_backoff * failed_requests, 15.0))

    async def evaluate_batch_image(images, prompts):
        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        async def _guarded(prompt, img):
            async with semaphore:
                return await evaluate_image(prompt, img)

        tasks = [_guarded(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        outputs = asyncio.run(evaluate_batch_image(images, prompts))
        text_outputs = [output[0] for output in outputs]
        failover_counts = [output[1] for output in outputs]
        scores, details = _extract_pointwise_scores(text_outputs)
        details["unifiedreward_failover_count"] = failover_counts
        return scores, details
    
    return _fn


def _images_to_uint8_nhwc(images):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        if images.ndim == 4 and images.shape[1] in (1, 3, 4):
            images = images.transpose(0, 2, 3, 1)
    return images


def _scalar_details(details):
    """Keep the single-valued numeric entries of one image's verifier details."""
    row = {}
    for key, value in details.items():
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                continue
            value = value[0]
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            row[key] = float(value)
    return row


# The soft gates multiply the verifier's partial-credit score (reward_mode="default"). They are the gates
# used to train the paper's models.
def _soft_gate(base_score, row):
    def get(key, default):
        value = row.get(key, default)
        return float(default) if value == _REWARD_MISSING_VALUE else float(value)

    gates = {
        "valid": get("visual_logic_valid_spec", 1.0),
        "count": 0.40 + 0.60 * get("visual_logic_object_count_score", 0.0),
        "shape": 0.40 + 0.60 * get("visual_logic_shape_score", 0.0),
        "temporal": 0.35 + 0.65 * get("visual_logic_temporal_score", 1.0),
        "detection": 0.50 + 0.50 * get("visual_logic_video_detection_rate", 1.0),
        "relation": 0.50 + 0.50 * get("visual_logic_relation_score", 1.0),
        "extra": 1.0 / (1.0 + 0.20 * max(0.0, get("visual_logic_extra_components", 0.0))),
        "forbidden": get("visual_logic_forbidden", 1.0),
    }
    score = float(base_score)
    for value in gates.values():
        score *= value
    return float(max(0.0, min(1.0, score))), gates


def vvr_score(device):
    """VVR training reward: the verifier's partial-credit score times the soft gates."""
    del device
    from vvr_bench.verifier import score_sample_spec

    def _fn(images, prompts, metadata):
        del prompts
        images = _images_to_uint8_nhwc(images)
        scores = []
        details = defaultdict(list)
        for image, meta in zip(images, metadata):
            spec = (meta or {}).get("visual_spec")
            if spec is None:
                raise ValueError("vvr_score needs metadata['visual_spec'] for every image")
            if isinstance(spec, str):
                spec = json.loads(spec)
            base_score, raw = score_sample_spec(np.asarray(image), spec)
            row = _scalar_details(raw)
            score, gates = _soft_gate(base_score, row)
            scores.append(score)
            details["vvr_base"].append(float(base_score))
            details["vvr_strict"].append(row.get("visual_logic_strict", 0.0))
            for name, value in gates.items():
                details[f"vvr_gate_{name}"].append(float(value))
        return scores, dict(details)

    return _fn


_BASE_REWARDS = {
    "vvr": vvr_score,
    "pickscore": pickscore_score,
    "ocr": ocr_score,
    "geneval": geneval_score,
    "geneval2": geneval2_score,
    "unifiedreward": unifiedreward_score,
}


def _call(name, fn, images, prompts, metadata):
    """Call one base reward and return (scores, details)."""
    if name == "geneval":
        scores, rewards, strict_rewards, _, _ = fn(images, prompts, metadata, True)
        return scores, {"geneval_accuracy": rewards, "geneval_strict_accuracy": strict_rewards}
    scores, details = fn(images, prompts, metadata)
    return scores, details or {}


def mixed_score(device, sources):
    """Score each image with the reward of its prompt's source.

    A prompt with a `visual_spec` gets the VVR reward; any other prompt gets the reward named by its
    metadata `training_source` (or `source`, or `tag`).
    """
    fns = {name: _BASE_REWARDS[name](device) for name in sorted(set(sources) | {"vvr"})}

    def _fn(images, prompts, metadata):
        groups = defaultdict(list)
        for index, meta in enumerate(metadata):
            meta = meta or {}
            if "visual_spec" in meta:
                source = "vvr"
            else:
                source = str(meta.get("training_source") or meta.get("source") or meta.get("tag"))
            if source not in fns:
                raise ValueError(f"no reward for source {source!r} (configured: {sorted(fns)})")
            groups[source].append(index)
        scores = [0.0] * len(metadata)
        details = {}
        for source, indices in groups.items():
            sub_scores, sub_details = _call(
                source,
                fns[source],
                _slice_batch_images(images, indices),
                [prompts[i] for i in indices],
                [metadata[i] for i in indices],
            )
            for local, original in enumerate(indices):
                scores[original] = float(sub_scores[local])
            _fill_subset_details(details, {source: list(sub_scores), **sub_details}, indices, len(metadata))
        for source in fns:
            details[f"is_{source}"] = [1.0 if i in groups.get(source, ()) else 0.0 for i in range(len(metadata))]
        return scores, details

    return _fn


def multi_score(device, score_dict, mixed_sources=None):
    """Build the training reward from `{reward_name: weight}`.

    Reward names are the keys of `_BASE_REWARDS`, or `"mixed"` with `mixed_sources` listing the non-VVR
    sources. The returned function gives a dict of per-image arrays whose `avg` entry is the weighted reward.
    """
    fns = {}
    for name in score_dict:
        if name == "mixed":
            fns[name] = mixed_score(device, mixed_sources or [])
        else:
            fns[name] = _BASE_REWARDS[name](device)

    def _fn(images, prompts, metadata, only_strict=True):
        del only_strict
        expected = len(prompts)
        total = np.zeros(expected, dtype=np.float64)
        details = {}
        for name, weight in score_dict.items():
            scores, sub_details = _call(name, fns[name], images, prompts, metadata)
            scores = _coerce_reward_scores(scores, expected, f"multi_score {name}")
            details[name] = scores
            for key, value in sub_details.items():
                vector = _coerce_numeric_vector(value, expected, key, "multi_score details")
                if vector is not None:
                    details[key] = vector
            total += float(weight) * np.asarray(scores, dtype=np.float64)
        details["avg"] = total.tolist()
        return details, {}

    return _fn
