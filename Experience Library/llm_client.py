"""
Lightweight LLM Client for inference.

Uses OpenAI API directly for text generation, and local Gemma 3 4B
for vision / image analysis (no OpenAI vision calls needed).

If the API key is not set, text generation falls back to a placeholder
that prints the prompt and returns a canned response (useful for
dry-run testing).
"""

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


class InferenceLLMClient:
    """
    Thin OpenAI wrapper for agent response generation.

    Resolution order for API key:
        1. Constructor argument
        2. OPENAI_API_KEY environment variable
        3. .env file in project root (via python-dotenv if available)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        max_tokens: int = 400,
        load_vlm: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        # Resolve API key
        # API key: env var or .env file — no hardcoded value.
        key = os.environ.get("OPENAI_API_KEY", "")

        if not key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                key = os.environ.get("OPENAI_API_KEY")
            except ImportError:
                pass

        self._local_vlm = None  # optional external VLM (legacy attach_local_vlm)

        # ── Local Gemma 3 4B for vision ──
        self._vlm_model = None
        self._vlm_processor = None
        self._vlm_device = "cpu"
        if load_vlm:
            self._init_gemma_vlm()

        if key and HAS_OPENAI:
            self.client = OpenAI(api_key=key)
            print(f"  [LLM] OpenAI client ready  (model={self.model})")
        else:
            reason = "openai package not installed" if not HAS_OPENAI else "API key not found"
            print(f"  [LLM] WARNING: {reason} → dry-run mode (prompts printed, no API calls)")

    # ------------------------------------------------------------------
    # GEMMA 3 4B VLM INITIALISATION
    # ------------------------------------------------------------------

    def _init_gemma_vlm(self):
        """Load the local Gemma 3 4B vision model (same as openai_client.py)."""
        if not HAS_VLM_DEPS:
            print("  [VLM] torch / transformers / PIL not installed → image analysis unavailable")
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
            print(f"  [VLM] Gemma 3 4B ready")
        except Exception as e:
            print(f"  [VLM] Failed to load Gemma VLM: {e}")
            self._vlm_model = None
            self._vlm_processor = None

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate a response from the LLM.

        Returns:
            (response_text, token_usage_dict)
        """
        temp = temperature or self.temperature
        mtok = max_tokens or self.max_tokens

        if self.client is None:
            return self._dry_run(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a persuasive motor insurance sales agent trying to persuade the user to buy motor insurance for their car. You genuinely believe this product protects the customer — your job is to make them believe it too. Persuade aggressively but naturally. Never sound desperate or scripted. Match the tone, length, and vocabulary of the Experience Library responses. Be concise."},
                    {"role": "user", "content": prompt},
                ],
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

    # ------------------------------------------------------------------

    def _dry_run(self, prompt: str) -> Tuple[str, Dict]:
        """Print prompt and return placeholder (no API call)."""
        print("\n" + "=" * 60)
        print("DRY-RUN PROMPT (no LLM call):")
        print("=" * 60)
        # Show first 1500 chars
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
    # IMAGE / VISION ANALYSIS
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_url: str,
        query: str = "Analyze this vehicle image for insurance assessment.",
        context: str = "",
    ) -> Tuple[str, Dict]:
        """
        Analyze an image using the local Gemma 3 4B vision model.

        Falls back to an externally-attached VLM (legacy attach_local_vlm),
        then to dry-run if nothing is available.
        """
        prompt_text = f"{context}\n\n{query}" if context else query

        # ── Try built-in Gemma VLM ──
        if self._vlm_model is not None and self._vlm_processor is not None:
            return self._gemma_analyze(image_url, prompt_text)

        # ── Legacy: externally-attached VLM ──
        if self._local_vlm is not None:
            try:
                return self._local_vlm.analyze_image(image_url, prompt_text)
            except Exception as e:
                print(f"  [LLM] External VLM failed: {e}")

        # ── Dry-run ──
        return self._dry_run(
            f"[IMAGE ANALYSIS]\nURL: {image_url}\nQuery: {query}"
        )

    def _gemma_analyze(self, image_url: str, prompt: str) -> Tuple[str, Dict]:
        """Run image analysis through the local Gemma 3 4B model."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        # Load image
        raw_image = None
        try:
            if image_url.startswith("http"):
                resp = requests.get(
                    image_url, stream=True, timeout=10, headers=headers,
                )
                resp.raise_for_status()
                raw_image = Image.open(resp.raw).convert("RGB")
            else:
                raw_image = Image.open(image_url).convert("RGB")
        except Exception as e:
            print(f"  [VLM] Could not load image {image_url}: {e}")
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

            # Clean artefacts (same as openai_client.py)
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

    # ------------------------------------------------------------------
    # LOCAL VLM ATTACHMENT (optional Gemma 3 4B)
    # ------------------------------------------------------------------

    def attach_local_vlm(self, vlm_client) -> None:
        """
        Attach a parent-project OpenAIClient that has ``analyze_image()``.
        When set, ``analyze_image()`` will prefer the local VLM.
        """
        self._local_vlm = vlm_client
        print("  [LLM] Local VLM attached for image analysis")
