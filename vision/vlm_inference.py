"""
Vision-Language Model inference via OpenVINO.

Loads the Phi-3.5-vision-instruct INT4 model once and provides
an analyze_screen() function that takes a screenshot + instruction
and returns the model's raw text response.
"""

from pathlib import Path

from PIL import Image

from utils.logger import get_logger

log = get_logger("vlm")

# Prompt template instructs the model to return a strict JSON action plan.
SYSTEM_PROMPT = """\
You are a desktop automation agent. Given a screenshot and a task, \
output ONLY a JSON object. No markdown fences, no explanation.

Format: {{"thought":"...","actions":[{{"type":"click","x":100,"y":200}}],\
"task_complete":false}}

Action types: click(x,y) | double_click(x,y) | type(text) | scroll(amount) | wait(seconds)
Coordinates are screen pixels. Set task_complete to true when the task is finished."""


class VLMInference:
    """Singleton-style wrapper around the OpenVINO vision model."""

    def __init__(self, model_path: str | Path, device: str = "CPU"):
        self.model_path = str(model_path)
        self.device = device
        self.model = None
        self.processor = None

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
        max_new_tokens: int = 1024,
    ) -> str:
        """Send screenshot + instruction to the VLM, return raw text output."""
        if not self.is_loaded:
            self.load()

        user_content = (
            "<|image_1|>\n"
            f"{SYSTEM_PROMPT}\n\n"
            f"USER TASK: {instruction}"
        )
        messages = [{"role": "user", "content": user_content}]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt")

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Slice off the prompt tokens so we only decode the new generation
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        log.info("VLM response (%d chars): %s", len(response), response[:200])
        return response.strip()
