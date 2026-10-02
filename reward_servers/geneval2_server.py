#!/usr/bin/env python3
"""HTTP reward server for GenEval2 Soft-TIFA scoring.

The training environment is pinned to an older transformers version for SD3.
Run this server with a newer transformers overlay that supports Qwen3-VL.
"""

import argparse
import math
import os
import pickle
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import torch
from PIL import Image


def return_numeric_string(number):
    mapping = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return mapping.get(str(number).lower(), "other")


def answer_variants(question, answer):
    if question.startswith("How many"):
        number = return_numeric_string(answer)
        variants = [answer, answer.capitalize(), " " + answer, " " + answer.capitalize()]
        if number != "other":
            variants.extend([number, " " + number])
        return variants
    return ["Yes", "yes", " yes", " Yes"]


def construct_message(question, image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


def first_token_ids(tokenizer, answers):
    ids = []
    for answer in answers:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if token_ids:
            ids.append(token_ids[0])
    return sorted(set(ids))


def score_question(question, answer, image):
    prompt = f"{question} Answer in one word."
    messages = construct_message(prompt, image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
    ids = first_token_ids(processor.tokenizer, answer_variants(question, answer))
    if not ids:
        return 0.0
    return float(probs[0, ids].sum().item())


def score_image(image, metadata):
    score_list = []
    for question, answer in metadata.get("vqa_list", []):
        score_list.append(score_question(question, answer, image))
    if not score_list:
        return 0.0, 0.0, []
    am = float(sum(score_list) / len(score_list))
    eps = 1e-8
    gm = float(math.exp(sum(math.log(max(s, eps)) for s in score_list) / len(score_list)))
    return am, gm, score_list


class GenEval2Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        data = pickle.loads(self.rfile.read(length))
        images_bytes = data["images"]
        metadatas = data["meta_datas"]
        method = data.get("method", DEFAULT_METHOD)

        am_scores = []
        gm_scores = []
        selected_scores = []
        question_counts = []
        atom_counts = []
        per_prompt_score_lists = []

        for img_bytes, meta in zip(images_bytes, metadatas):
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            am, gm, score_list = score_image(image, meta or {})
            am_scores.append(am)
            gm_scores.append(gm)
            selected_scores.append(am if method == "soft_tifa_am" else gm)
            question_counts.append(len(score_list))
            atom_counts.append(int((meta or {}).get("atom_count", len(score_list))))
            per_prompt_score_lists.append(score_list)

        response = {
            "scores": selected_scores,
            "geneval2_am": am_scores,
            "geneval2_gm": gm_scores,
            "geneval2_question_count": question_counts,
            "geneval2_atom_count": atom_counts,
            "score_lists": per_prompt_score_lists,
        }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(pickle.dumps(response))

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        if args and "200" not in str(args[0]):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="GenEval2 Soft-TIFA server")
    parser.add_argument("--port", type=int, default=18185)
    parser.add_argument("--model-name", type=str, default=os.environ.get("GENEVAL2_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--device", type=str, default=os.environ.get("GENEVAL2_DEVICE", "cuda"))
    parser.add_argument("--method", type=str, default=os.environ.get("GENEVAL2_METHOD", "soft_tifa_gm"),
                        choices=["soft_tifa_am", "soft_tifa_gm"])
    args = parser.parse_args()

    global processor, model, DEFAULT_METHOD
    DEFAULT_METHOD = args.method

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    t0 = time.time()
    print(f"[GenEval2] Loading processor/model {args.model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name, torch_dtype="auto")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=None,
    )
    model = model.to(args.device).eval()
    print(f"[GenEval2] Model loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"[GenEval2] Server listening on 0.0.0.0:{args.port}, method={args.method}", flush=True)
    HTTPServer(("0.0.0.0", args.port), GenEval2Handler).serve_forever()


if __name__ == "__main__":
    main()
