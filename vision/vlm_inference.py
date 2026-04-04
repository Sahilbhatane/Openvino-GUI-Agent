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

Execute EXACTLY ONE action per step. Output ONLY one valid JSON object. No markdown, no code fences, no extra text.

Example shape (replace values; use a single leading "{" and trailing "}" only):
{"thought":"brief reasoning","action":{"type":"click","element":3},"task_complete":false}

Action types (use ONLY these shapes):
  click        -- {"type":"click","element":<id>}  or  {"type":"click","x":<int>,"y":<int>}
  double_click -- {"type":"double_click","element":<id>}
  type         -- {"type":"type","text":"<string>","element":<id>}
  scroll       -- {"type":"scroll","amount":<int>}  (positive=up, negative=down)
  wait         -- {"type":"wait","seconds":<number>}
  press_key    -- {"type":"press_key","key":"<name>"}  (enter, escape, tab, space, backspace, delete, up, down, left, right, win, f1-f12)
  hotkey       -- {"type":"hotkey","keys":["ctrl","c"]}

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


def _log_openvino_runtime_devices(visual_model, requested_device: str) -> None:
    """Log devices OpenVINO exposes and, when possible, where compiled subgraphs run."""
    try:
        import openvino as ov

        core = ov.Core()
        log.info("OpenVINO Core available_devices=%s", list(core.available_devices))
    except Exception as exc:
        log.warning("Could not query OpenVINO Core().available_devices: %s", exc)

    target = getattr(visual_model, "_device", None) or requested_device
    log.info("OpenVINO model compile target string: %s (config/request was %s)", target, requested_device)

    hints: list[str] = []

    def _execution_devices_for_request(req) -> str | None:
        if req is None:
            return None
        candidates = [req]
        gcm = getattr(req, "get_compiled_model", None)
        if callable(gcm):
            try:
                candidates.append(gcm())
            except Exception:
                pass
        for cand in candidates:
            if cand is None or not hasattr(cand, "get_property"):
                continue
            try:
                devs = cand.get_property("EXECUTION_DEVICES")
                if devs:
                    return str(devs)
            except Exception:
                continue
        return None

    def _append_exec(label: str, req) -> None:
        devs = _execution_devices_for_request(req)
        if devs:
            hints.append(f"{label}={devs}")

    try:
        ve = getattr(visual_model, "vision_embeddings", None)
        if ve is not None:
            _append_exec("vision_embeddings", getattr(ve, "request", None))
    except Exception as exc:
        log.debug("vision_embeddings EXECUTION_DEVICES: %s", exc)

    try:
        lm = getattr(visual_model, "language_model", None)
        if lm is not None:
            _append_exec("language_model", getattr(lm, "request", None))
    except Exception as exc:
        log.debug("language_model EXECUTION_DEVICES: %s", exc)

    if hints:
        log.info("OpenVINO compiled subgraph EXECUTION_DEVICES: %s", "; ".join(hints))
    else:
        log.info(
            "Tip: run `python -c \"import openvino as ov; print(ov.Core().available_devices)\"` "
            "to see devices; use GPU/NPU in config only if listed (OpenVINO does not use NVIDIA CUDA)."
        )


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

        cache_dir = str(Path(self.model_path).parent / ".ov_cache")
        ov_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": cache_dir,
            "KV_CACHE_PRECISION": "u8",
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
        }

        log.info("Loading OpenVINO model (device=%s) ...", self.device)
        from optimum.intel.openvino import OVModelForVisualCausalLM

        self.model = OVModelForVisualCausalLM.from_pretrained(
            self.model_path, device=self.device, trust_remote_code=True,
            ov_config=ov_config,
        )
        log.info("Model loaded successfully.")
        _log_openvino_runtime_devices(self.model, self.device)

    def warmup(self) -> None:
        """Run a tiny inference so the first real call doesn't pay cold-start cost."""
        if not self.is_loaded:
            return
        log.info("Warming up model (single short generate) ...")
        dummy = Image.new("RGB", (64, 64), color=(0, 0, 0))
        try:
            self.analyze_screen(dummy, "warmup", max_new_tokens=1)
        except Exception as exc:
            log.warning("Warmup generate failed (non-fatal): %s", exc)
        self.last_usage = None
        log.info("Warmup complete.")

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
