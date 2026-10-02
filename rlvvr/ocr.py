import os
import multiprocessing as mp
import traceback
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union, Tuple
from PIL import Image
import filelock
import shutil
import tempfile
import threading

# Serialize only shared-cache preparation. PaddleOCR construction itself must
# not sit under this lock: many DDP ranks initialize OCR workers concurrently.
_PADDLE_LOCK = os.path.join(tempfile.gettempdir(), "paddleocr_init.lock")
_DEFAULT_PADDLEX_HOME = os.path.join(os.path.expanduser("~"), ".cache", "paddlex")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _prepare_paddle_env(force_cpu: bool = False):
    # Disable MKL-DNN to avoid PIR model inference errors on CPU.
    os.environ.setdefault('FLAGS_use_mkldnn', '0')
    os.environ.setdefault('FLAGS_use_onednn', '0')
    os.environ.setdefault('FLAGS_enable_pir_api', '0')
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    if force_cpu:
        # Keep OCR out of the SD3 training GPUs. This runs before importing
        # Paddle in the OCR child/preflight process.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("FLAGS_selected_gpus", "")


def _init_paddleocr(force_cpu: bool = False):
    """Initialize PaddleOCR with a per-process copy of the model cache
    to avoid NFS read corruption when 64 processes read simultaneously."""
    _prepare_paddle_env(force_cpu=force_cpu)

    shared_cache = os.environ.get("OCR_PADDLEX_HOME", _DEFAULT_PADDLEX_HOME)
    if shared_cache:
        os.makedirs(os.path.dirname(shared_cache), exist_ok=True)
        home_cache = os.path.expanduser("~/.paddlex")
        if not os.path.isdir(shared_cache) and os.path.isdir(home_cache):
            with filelock.FileLock(_PADDLE_LOCK, timeout=120):
                if not os.path.isdir(shared_cache):
                    shutil.copytree(home_cache, shared_cache)

    proc_cache = os.path.join(tempfile.gettempdir(), f"paddleocr_cache_{os.getpid()}")
    if shared_cache and os.path.isdir(shared_cache) and not os.path.isdir(proc_cache):
        shutil.copytree(shared_cache, proc_cache)
    if os.path.isdir(proc_cache):
        os.environ["PADDLEX_HOME"] = proc_cache
        os.environ["PADDLE_PDX_CACHE_HOME"] = proc_cache
    elif shared_cache:
        if not os.path.isdir(shared_cache):
            os.makedirs(shared_cache, exist_ok=True)
        os.environ["PADDLEX_HOME"] = shared_cache
        os.environ["PADDLE_PDX_CACHE_HOME"] = shared_cache

    import paddle
    from paddleocr import PaddleOCR

    paddle.set_flags({'FLAGS_use_mkldnn': False})
    ocr = PaddleOCR(use_angle_cls=False, lang="en", enable_mkldnn=False)
    return ocr


def _extract_text(ocr, img):
    """Run OCR and extract recognized text from PaddleOCR predict() result."""
    results = ocr.predict(img)
    text = ''
    if results:
        r = results[0]
        if r.get('rec_texts'):
            text = ''.join(
                t for t, s in zip(r['rec_texts'], r['rec_scores']) if s > 0
            )
    return text


class _DirectOcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        force_cpu_direct = (not use_gpu) and _truthy_env("OCR_FORCE_CPU_DIRECT", "0")
        self.ocr = _init_paddleocr(force_cpu=force_cpu_direct)
        self._ocr_lock = threading.Lock()

    @torch.no_grad()
    def __call__(self,
                images: Union[List[Image.Image], List[np.ndarray]],
                prompts: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] if '"' in prompt else prompt for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition
                with self._ocr_lock:
                    recognized_text = _extract_text(self.ocr, img)

                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1-dist/(len(prompt))
            rewards.append(reward)

        return rewards


def _ocr_worker_main(conn):
    os.environ["OCR_ISOLATE_WORKER"] = "0"
    _prepare_paddle_env(force_cpu=not _truthy_env("OCR_WORKER_USE_GPU", "0"))
    scorer = _DirectOcrScorer()
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        if not message:
            continue
        command = message[0]
        if command == "close":
            break
        if command != "score":
            conn.send(("error", f"unknown OCR worker command: {command!r}"))
            continue
        _, images, prompts = message
        try:
            conn.send(("ok", scorer(images, prompts)))
        except BaseException:
            conn.send(("error", traceback.format_exc()))


class _IsolatedOcrScorer:
    """Run PaddleOCR in a child process so native aborts do not kill DDP ranks."""

    def __init__(self, use_gpu: bool = False):
        self._ctx = mp.get_context("spawn")
        self._timeout_s = float(os.environ.get("OCR_WORKER_TIMEOUT", "600"))
        self._max_restarts = int(os.environ.get("OCR_WORKER_RESTARTS", "1"))
        self._conn = None
        self._proc = None
        self._start_worker()

    def _start_worker(self):
        self._stop_worker()
        parent_conn, child_conn = self._ctx.Pipe()
        proc = self._ctx.Process(target=_ocr_worker_main, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()
        self._conn = parent_conn
        self._proc = proc

    def _stop_worker(self):
        if self._conn is not None:
            try:
                self._conn.send(("close",))
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5)
            self._proc = None

    def __del__(self):
        try:
            self._stop_worker()
        except Exception:
            pass

    @torch.no_grad()
    def __call__(self,
                images: Union[List[Image.Image], List[np.ndarray]],
                prompts: List[str]) -> torch.Tensor:
        last_error = None
        for attempt in range(self._max_restarts + 1):
            if self._proc is None or not self._proc.is_alive():
                self._start_worker()
            try:
                self._conn.send(("score", images, prompts))
                if not self._conn.poll(self._timeout_s):
                    raise TimeoutError(f"OCR worker timed out after {self._timeout_s}s")
                status, payload = self._conn.recv()
                if status == "ok":
                    return payload
                raise RuntimeError(payload)
            except Exception as exc:
                last_error = exc
                print(f"OCR worker failed on attempt {attempt + 1}: {exc}")
                self._start_worker()
        raise RuntimeError(
            f"OCR worker failed after {self._max_restarts + 1} attempt(s)"
        ) from last_error


class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        if _truthy_env("OCR_ISOLATE_WORKER"):
            self._impl = _IsolatedOcrScorer(use_gpu=use_gpu)
        else:
            self._impl = _DirectOcrScorer(use_gpu=use_gpu)

    @torch.no_grad()
    def __call__(self,
                images: Union[List[Image.Image], List[np.ndarray]],
                prompts: List[str]) -> torch.Tensor:
        return self._impl(images, prompts)

class OcrScorer_video_or_image:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = _init_paddleocr()
        self._ocr_lock = threading.Lock()
        self.frame_interval = 4

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> Tuple[List[float], torch.Tensor]:
        """
        :param images: List of images or videos (each video as np.ndarray of shape [F, H, W, C])
        :param prompts: List of prompts containing target text
        :return: (List of OCR rewards, Tensor of attention regions)
        """
        prompts = [prompt.split('"')[1] if '"' in prompt else prompt for prompt in prompts]
        assert len(images) == len(prompts), "Mismatch between images and prompts."

        rewards = []
        for img, prompt in zip(images, prompts):
            prompt = prompt.replace(' ', '').lower()
            frame_rewards = []

            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
            else:
                sampled_frames = [img]

            for frame in sampled_frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    with self._ocr_lock:
                        text = _extract_text(self.ocr, frame)
                    text = text.replace(' ', '').lower()

                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))

                except Exception as e:
                    print(f"OCR failed on frame: {e}")
                    dist = len(prompt)

                reward = 1 - dist / len(prompt)
                if reward > 0:
                    frame_rewards.append(reward)

            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)

        return rewards

if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")
