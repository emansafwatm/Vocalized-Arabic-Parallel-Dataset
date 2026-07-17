"""Unit tests for partial Arabic vocalization."""

from pathlib import Path
import importlib.util
import sys

SCRIPT = Path(__file__).parents[1] / "scripts" / "create_partial_vocalization.py"
spec = importlib.util.spec_from_file_location("partial_vocalization", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
remove_word_final_endings = module.remove_word_final_endings


def test_removes_final_short_vowels_and_tanween() -> None:
    assert remove_word_final_endings("الطَّالِبُ مُجْتَهِدٌ") == "الطَّالِب مُجْتَهِد"


def test_preserves_internal_diacritics() -> None:
    assert remove_word_final_endings("مَدْرَسَةٌ") == "مَدْرَسَة"


def test_preserves_terminal_shadda() -> None:
    assert remove_word_final_endings("حَقٌّ") == "حَقّ"
    assert remove_word_final_endings("مَرّ") == "مَرّ"


def test_preserves_punctuation() -> None:
    assert remove_word_final_endings("مَرْحَبًا، بِكُمْ!") == "مَرْحَبا، بِكُم!"


def test_non_arabic_tokens_unchanged() -> None:
    assert remove_word_final_endings("Version 2.0") == "Version 2.0"
