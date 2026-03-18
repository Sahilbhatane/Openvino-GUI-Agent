"""
Vision-Language Model inference via OpenVINO.

Loads the Phi-3.5-vision-instruct INT4 model once and provides
an analyze_screen() function that takes a screenshot + instruction
and returns the model's raw text response.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from utils.logger import get_logger

log = get_logger("vlm")

SYSTEM_PROMPT = """\
You are a desktop automation agent operating step-by-step. You see a screenshot \
with numbered red badges marking interactive elements, plus a text list mapping \
each number to an element.

Execute EXACTLY ONE action per step. Output ONLY a JSON object. No markdown, no explanation.

Format: {{"thought":"brief reasoning about what to do next","action":{{...}},"task_complete":false}}

Action types (use ONLY these):
  click        -- {{"type":"click","element":<id>}}  or  {{"type":"click","x":<int>,"y":<int>}}
  double_click -- {{"type":"double_click","element":<id>}}
  type         -- {{"type":"type","text":"<string>","element":<id>}}
  scroll       -- {{"type":"scroll","amount":<int>}}  (positive=up, negative=down)
  wait         -- {{"type":"wait","seconds":<number>}}
  press_key    -- {{"type":"press_key","key":"<name>"}}  (enter, escape, tab, space, backspace, delete, up, down, left, right, win, f1-f12)
  hotkey       -- {{"type":"hotkey","keys":["ctrl","c"]}}  (key combination)

Rules:
- Return EXACTLY ONE action per response. Never return a list of actions.
- Use "element" to reference numbered UI elements. Fall back to x,y only if no element matches.
- To open an app: click the search bar, type the name, press enter.
- Set task_complete to true (with no action) when the task is finished.
- If the screen hasn't changed after your last action, try a different approach."""


@dataclass
class TokenUsage:
    """Tracks token counts and latency for a single VLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time: float = 0.0  # seconds

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.generation_time if self.generation_time > 0 else 0.0


class VLMInference:
    """Singleton-style wrapper around the OpenVINO vision model."""

    def __init__(self, model_path: str | Path, device: str = "CPU"):
        self.model_path = str(model_path)
        self.device = device
        self.model = None
        self.processor = None
        self.last_usage: TokenUsage | None = None

    # ── loading ───────────────────────────────────────────

    def load(self) -> None:
        """Load model + processor into memory (slow -- call once at startup)."""
        log.info("Loading processor from %s ...", self.model_path)
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        log.info("Loading OpenVINO model (device=%s) ...", self.device)
        from optimum.intel.openvino import OVModelForVisualCausalLM

        self.model = OVModelForVisualCausalLM.from_pretrained(
            self.model_path, device=self.device, trust_remote_code=True
        )
        log.info("Model loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    # ── inference ─────────────────────────────────────────

    def analyze_screen(
        self,
        image: Image.Image,
        instruction: str,
        elements_text: str = "",
        history_text: str = "",
        max_new_tokens: int = 300,
    ) -> str:
        """Send screenshot + instruction to the VLM, return raw text output."""
        if not self.is_loaded:
            self.load()

        history_block = f"\n\n{history_text}" if history_text else ""
        element_block = f"\n\n{elements_text}" if elements_text else ""
        user_content = (
            "<|image_1|>\n"
            f"{SYSTEM_PROMPT}"
            f"{history_block}"
            f"{element_block}\n\n"
            f"USER TASK: {instruction}"
        )
        messages = [{"role": "user", "content": user_content}]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt")
        input_token_count = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_time = time.perf_counter() - t0

        new_tokens = output_ids[:, input_token_count:]
        output_token_count = new_tokens.shape[1]

        self.last_usage = TokenUsage(
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            generation_time=gen_time,
        )

        response = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        log.info(
            "Tokens: %d in / %d out  (%.1fs, %.1f tok/s) | response: %s",
            input_token_count,
            output_token_count,
            gen_time,
            self.last_usage.tokens_per_second,
            response[:160],
        )
        return response.strip()
