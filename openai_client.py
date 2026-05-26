"""
Unified OpenAI + local VLM client for dataset generation.

Text generation and judge calls go through OpenAI. Vision analysis
uses a local Gemma 3 4B VLM when available (paper Appendix B.3).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import torch
    from PIL import Image
    import requests
    from transformers import AutoProcessor, AutoModelForImageTextToText
    HAS_VLM_DEPS = True
except ImportError:
    HAS_VLM_DEPS = False


class OpenAIClient:
    """
    Minimal client for text + judge + vision.

    Defaults follow the paper:
      - text/judge model: gpt-5
      - vision: local Gemma 3 4B IT
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        temperature: float = 0.8,
        max_tokens: int = 400,
        load_vlm: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                key = os.environ.get("OPENAI_API_KEY", "")
            except ImportError:
                pass

        if key and HAS_OPENAI:
            self.client = OpenAI(api_key=key)
            print(f"  [LLM] OpenAI client ready (model={self.model})")
        else:
            reason = "openai package not installed" if not HAS_OPENAI else "API key not found"
            print(f"  [LLM] WARNING: {reason} -> dry-run mode")

        # Local VLM
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_device = "cpu"
        if load_vlm:
            self._init_gemma_vlm()

    # ------------------------------------------------------------------
    # TEXT GENERATION
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict]:
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        if self.client is None:
            return self._dry_run(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=mtok,
            )
            text = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return text, usage
        except Exception as e:
            print(f"  [LLM] API error: {e}")
            return self._dry_run(prompt)

    def judge_turn(self, prompt: str) -> Tuple[Dict, Dict]:
        """Run a judge prompt and parse JSON output."""
        text, usage = self.generate_text(prompt, temperature=0.2, max_tokens=800)
        try:
            result = json.loads(text)
            return result, usage
        except Exception:
            return {"error": "judge_json_parse_failed", "raw": text}, usage

    def _dry_run(self, prompt: str) -> Tuple[str, Dict]:
        print("\n" + "=" * 60)
        print("DRY-RUN PROMPT (no LLM call):")
        print("=" * 60)
        print(prompt[:1500])
        if len(prompt) > 1500:
            print(f"\n... ({len(prompt) - 1500} more chars) ...")
        print("=" * 60 + "\n")

        return (
            "[DRY-RUN] This is a placeholder response. "
            "Set OPENAI_API_KEY to enable real generation.",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # ------------------------------------------------------------------
    # VISION (Gemma 3 4B IT)
    # ------------------------------------------------------------------

    def _init_gemma_vlm(self) -> None:
        if not HAS_VLM_DEPS:
            print("  [VLM] torch/transformers/PIL not installed -> image analysis unavailable")
            return

        vlm_model_id = "google/gemma-3-4b-it"
        self._vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [VLM] Loading {vlm_model_id} on {self._vlm_device} ...")
        try:
            self._vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
            self._vlm_model = AutoModelForImageTextToText.from_pretrained(
                vlm_model_id,
                device_map="auto",
            )
            print("  [VLM] Gemma 3 4B ready")
        except Exception as e:
            print(f"  [VLM] Failed to load Gemma VLM: {e}")
            self._vlm_model = None
            self._vlm_processor = None

    def analyze_image(self, image_url: str, prompt: str) -> Tuple[str, Dict]:
        """Analyze an image using the local Gemma 3 4B model."""
        if self._vlm_model is None or self._vlm_processor is None:
            return self._dry_run(f"[IMAGE ANALYSIS]\nURL: {image_url}\nPrompt: {prompt}")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        try:
            if image_url.startswith("http"):
                resp = requests.get(image_url, stream=True, timeout=10, headers=headers)
                resp.raise_for_status()
                raw_image = Image.open(resp.raw).convert("RGB")
            else:
                raw_image = Image.open(image_url).convert("RGB")
        except Exception as e:
            return (
                f"[Image load error: {e}]",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text_prompt = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._vlm_processor(
                text=text_prompt, images=raw_image, return_tensors="pt",
            ).to(self._vlm_model.device)

            with torch.no_grad():
                outputs = self._vlm_model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.7,
                    do_sample=True,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            response_text = self._vlm_processor.decode(
                outputs[0][prompt_len:], skip_special_tokens=True,
            ).strip()

            # Clean common artefacts
            for tag in ("Assistant:", "<end_of_turn>", "User:"):
                if tag in response_text:
                    response_text = response_text.split(tag)[-1].strip()

            token_usage = {
                "prompt_tokens": prompt_len,
                "completion_tokens": int(outputs.shape[-1]) - prompt_len,
                "total_tokens": int(outputs.shape[-1]),
            }
            return response_text, token_usage

        except Exception as e:
            print(f"  [VLM] Generation error: {e}")
            return (
                f"[VLM generation error: {e}]",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )
