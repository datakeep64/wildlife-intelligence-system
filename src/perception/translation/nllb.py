"""NLLB-200 (distilled 600M) translation of sighting alerts.

Two modes, to measure a known MT weakness - named entities:
  raw       translate the full English alert
  glossary  replace species/place names with placeholder tokens, translate,
            then insert names from the ontology's multilingual labels.
            If the model mangles a placeholder, we fall back to raw and log it.

~2.4 GB weights (fp32). Runs on CPU (slow-ish, ~1-3 s/sentence) or Apple MPS.
"""
import re
from functools import lru_cache

import torch

from src.common.paths import device as best_device

MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {"ms": "zsm_Latn", "ta": "tam_Taml", "zh": "zho_Hans", "en": "eng_Latn"}


class Translator:
    def __init__(self, model: str = MODEL, device: str | None = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, src_lang=LANGS["en"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.device = device or best_device()
        self.model.to(self.device).eval()

    @torch.no_grad()
    def translate(self, texts: list[str], tgt: str, src: str = "en", max_new_tokens: int = 200) -> list[str]:
        self.tok.src_lang = LANGS[src]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **enc,
            forced_bos_token_id=self.tok.convert_tokens_to_ids(LANGS[tgt]),
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )
        return self.tok.batch_decode(out, skip_special_tokens=True)


@lru_cache(maxsize=1)
def get_translator() -> Translator:
    return Translator()


# ---------------------------------------------------------- glossary mode
def protect(text: str, terms: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Replace each English term with a placeholder like 'X1'. Returns (text, {placeholder: term})."""
    mapping = {}
    for i, term in enumerate(sorted(terms, key=len, reverse=True), start=1):
        ph = f"X{i}"
        if term in text:
            text = text.replace(term, ph)
            mapping[ph] = term
    return text, mapping


def restore(text: str, mapping: dict[str, str], terms: dict[str, str]) -> tuple[str, bool]:
    """Put target-language names back. Returns (text, all_placeholders_survived)."""
    ok = True
    for ph, term in mapping.items():
        if re.search(rf"\b{ph}\b", text):
            text = re.sub(rf"\b{ph}\b", terms[term], text)
        else:
            ok = False
    return text, ok


def translate_alert(text: str, tgt: str, glossary: dict[str, str] | None = None, translator: Translator | None = None) -> dict:
    """glossary: {English term: target-language term}. None -> raw mode."""
    tr = translator or get_translator()
    raw = tr.translate([text], tgt)[0]
    result = {"src": text, "tgt_lang": tgt, "raw": raw, "glossary": None, "placeholders_ok": None}
    if glossary:
        prot, mapping = protect(text, glossary)
        out = tr.translate([prot], tgt)[0]
        restored, ok = restore(out, mapping, glossary)
        result.update(glossary=restored if ok else raw, placeholders_ok=ok)
    return result
