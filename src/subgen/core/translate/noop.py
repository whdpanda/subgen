from subgen.core_types import Transcript, Segment
from subgen.core.translate.base import TranslatorProvider
from subgen.utils.logger import get_logger

logger = get_logger()

class NoopTranslator(TranslatorProvider):
    def translate(self, transcript: Transcript, target_lang: str = "zh") -> Transcript:
        logger.info("[Translate placeholder] returning mock zh text")
        zh_segments = []
        for s in transcript.segments:
            zh_segments.append(Segment(
                start=s.start, end=s.end,
                text="(翻译占位：这里将由小语种->中文引擎输出)",
                confidence=s.confidence
            ))
        return Transcript(language="zh", segments=zh_segments)
