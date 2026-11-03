import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class VLMJudge:
    """Client for the VLM judge. Builds the re-ranking messages and sends them to
    the vLLM server's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        max_tokens: int = 300,
        temperature: float = 0,
        top_p: float = 0.2,
        img_type: str = "image/jpeg",
        label_banner_h: int = 32,
        **kwargs,
    ):
        if not base_url:
            raise ValueError(f"No base_url for {model!r}, set it in configs/vlm/<name>.yaml")
        # Large multimodal requests can be slow on a local server
        timeout = httpx.Timeout(timeout=180.0, connect=10.0)
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = 0
        self.presence_penalty = kwargs.get("presence_penalty", 0)
        # vLLM sampling options with no OpenAI equivalent, sent through extra_body
        self.top_k = kwargs.get("top_k", None)
        self.min_p = kwargs.get("min_p", None)
        self.repetition_penalty = kwargs.get("repetition_penalty", None)
        self.enable_thinking = kwargs.get("enable_thinking", False)
        self.img_type = img_type
        self.label_banner_h = label_banner_h

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    @staticmethod
    def _fit_font(text: str, max_w: int, max_h: int, font_path: str) -> ImageFont.FreeTypeFont:
        """Largest font size (down to 8 px) that fits *text* in max_w x max_h.

        Falls back to PIL's default bitmap font if *font_path* can't be loaded.
        """
        size = max_h
        try:
            font = ImageFont.truetype(font_path, size)
        except OSError:
            return ImageFont.load_default()
        while size > 8:
            font = ImageFont.truetype(font_path, size)
            box = font.getbbox(text)
            if box[2] - box[0] <= max_w * 0.9 and box[3] - box[1] <= max_h * 0.8:
                return font
            size -= 2
        return font

    def label_image(self, path: str, label: str, font_path: str = _LABEL_FONT_PATH) -> Image.Image:
        """The image at *path* with a white banner holding *label* ("Query" or a
        candidate number) above it, so each crop carries its role in the pixels."""
        banner_h = self.label_banner_h
        im = Image.open(path).convert("RGB")
        font = self._fit_font(label, im.width, banner_h, font_path)
        out = Image.new("RGB", (im.width, banner_h + im.height), (255, 255, 255))
        out.paste(im, (0, banner_h))
        d = ImageDraw.Draw(out)
        d.text((im.width // 2, banner_h // 2), label, font=font, fill=(0, 0, 0), anchor="mm")
        return out

    def _image_to_data_url(self, path: str, label: Optional[str] = None) -> str:
        """Base64 data URL of an image file, with a label banner if *label* is given."""
        if label is not None:
            buf = io.BytesIO()
            self.label_image(path, label).save(buf, format="JPEG", quality=95)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        ext = Path(path).suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, self.img_type)
        data = base64.b64encode(Path(path).read_bytes()).decode()
        return f"data:{mime};base64,{data}"

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def build_rerank_prompt(
        self,
        query_path: str,
        candidate_paths: List[str],
        sys_text: str,
        user_text: str,
        user_prompt_position: str = "before",
        label_images: bool = False,
    ) -> list:
        """Messages for one re-ranking query: the Query and each candidate as a
        separate image entry, each preceded by a short caption.

        *user_prompt_position* puts the user instructions before ('before') or
        after ('after') the images. *label_images* burns the role into each image.
        """
        query_label = "Query" if label_images else None
        images: list = [
            {"type": "text", "text": "This is the Query."},
            {"type": "image_url", "image_url": {"url": self._image_to_data_url(query_path, label=query_label)}},
        ]
        for i, cpath in enumerate(candidate_paths):
            cand_label = str(i + 1) if label_images else None
            images.append({"type": "text", "text": f"Candidate {i + 1}."})
            images.append({"type": "image_url", "image_url": {"url": self._image_to_data_url(cpath, label=cand_label)}})
        user_part = {"type": "text", "text": user_text}
        content = [user_part] + images if user_prompt_position == "before" else images + [user_part]
        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": content},
        ]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def _build_api_params(self, messages: List[dict]) -> Dict[str, Any]:
        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if "qwen3" in self.model.lower():
            vllm_extra: Dict[str, Any] = {}
            # Qwen3.5/3.6 take a chat-template switch for thinking tokens
            if "qwen3.5" in self.model.lower() or "qwen3.6" in self.model.lower():
                vllm_extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            if self.top_k is not None:
                vllm_extra["top_k"] = self.top_k
            if self.min_p is not None:
                vllm_extra["min_p"] = self.min_p
            if self.repetition_penalty is not None:
                vllm_extra["repetition_penalty"] = self.repetition_penalty
            api_params["extra_body"] = vllm_extra
        return api_params

    async def chat_async(self, messages: List[dict]) -> Tuple[str, Any]:
        """Send a message list and return (reply text, token usage)."""
        response = await self.client.chat.completions.create(**self._build_api_params(messages))
        logger.debug("Raw response: %s", response)
        content = response.choices[0].message.content
        return (content.strip() if content is not None else ""), response.usage
