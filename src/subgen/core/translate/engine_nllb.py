from __future__ import annotations

from typing import Dict, Optional
from subgen.core.translate.base import TranslatorProvider
from subgen.core_types import Transcript, Segment
from subgen.utils.logger import get_logger

logger = get_logger()


# NLLB 语言代码示例（可持续补充）
# 你可以只先覆盖你常用的小语种
NLLB_LANG_MAP: Dict[str, str] = {
    # Auto handling: we will try to map detected lang
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "tr": "tur_Latn",
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
}


class NLLBTranslator(TranslatorProvider):
    """
    Local translation using Meta NLLB.
    Good coverage for low-resource languages.

    Default model choice:
      - facebook/nllb-200-distilled-600M (balanced)
      - You can switch to a larger model later for better quality.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers sentencepiece"
            ) from e

        import torch

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading NLLB model={model_name}, device={self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _map_lang(self, lang: str) -> str:
        key = (lang or "unknown").lower()
        return NLLB_LANG_MAP.get(key, "eng_Latn")  # fallback

    def translate(self, transcript: Transcript, target_lang: str = "zh") -> Transcript:
        import torch

        src_lang = self._map_lang(transcript.language)
        tgt_lang = self._map_lang(target_lang)

        # NLLB uses tokenizer language settings
        self.tokenizer.src_lang = src_lang
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        new_segments = []
        for s in transcript.segments:
            text = (s.text or "").strip()
            if not text:
                continue

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,        # 精度优先
                    length_penalty=1.0,
                )

            zh = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

            new_segments.append(
                Segment(
                    start=s.start,
                    end=s.end,
                    text=zh,
                    confidence=s.confidence,  # 暂沿用ASR置信度
                )
            )

        return Transcript(language=target_lang, segments=new_segments)
