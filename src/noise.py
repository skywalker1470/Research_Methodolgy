"""Character-level noise functions for robustness evaluation.

Each function is applied word-by-word so that noise stays within word
boundaries (spaces are never touched). Each character is perturbed
independently with probability `p`.
"""
import random
from typing import Callable, List

QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}


def _char_swap_word(word: str, p: float, rng: random.Random) -> str:
    chars = list(word)
    i = 0
    while i < len(chars) - 1:
        if rng.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    return "".join(chars)


def _char_delete_word(word: str, p: float, rng: random.Random) -> str:
    return "".join(
        c for c in word if not (c.isalpha() and rng.random() < p)
    )


def _keyboard_neighbor_word(word: str, p: float, rng: random.Random) -> str:
    chars = list(word)
    for i, c in enumerate(chars):
        if c.isalpha() and rng.random() < p:
            neighbors = QWERTY_NEIGHBORS.get(c.lower())
            if neighbors:
                repl = rng.choice(neighbors)
                chars[i] = repl.upper() if c.isupper() else repl
    return "".join(chars)


def _apply_word_fn(text: str, word_fn: Callable, p: float, rng: random.Random) -> str:
    return " ".join(word_fn(w, p, rng) for w in text.split(" "))


def char_swap(text: str, p: float, rng: random.Random) -> str:
    return _apply_word_fn(text, _char_swap_word, p, rng)


def char_delete(text: str, p: float, rng: random.Random) -> str:
    return _apply_word_fn(text, _char_delete_word, p, rng)


def keyboard_neighbor(text: str, p: float, rng: random.Random) -> str:
    return _apply_word_fn(text, _keyboard_neighbor_word, p, rng)


NOISE_FUNCS = {
    "swap": char_swap,
    "deletion": char_delete,
    "keyboard": keyboard_neighbor,
}
_NOISE_TYPE_INDEX = {"swap": 0, "deletion": 1, "keyboard": 2}

INTENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25]


def _combined_seed(seed: int, noise_type: str, p: float) -> int:
    # Python's hash() of str is salted per-process (PYTHONHASHSEED), so a
    # plain (seed, noise_type, p) tuple would not reproduce across runs.
    return seed * 1_000_003 + _NOISE_TYPE_INDEX[noise_type] * 1009 + round(p * 100)


def apply_noise(texts: List[str], noise_type: str, p: float, seed: int) -> List[str]:
    rng = random.Random(_combined_seed(seed, noise_type, p))
    fn = NOISE_FUNCS[noise_type]
    return [fn(t, p, rng) for t in texts]
