"""Find-S (Lab 3): learn the most specific hypothesis that fits every positive example."""


def find_s(examples: list[tuple[str, ...]], positive: list[bool]) -> tuple[tuple[str, ...] | None, list[tuple[int, tuple[str, ...]]]]:
    """Return the final hypothesis and a log of (example index, hypothesis after that change).

    The hypothesis starts as None (the most specific one, which fits nothing). The first positive example is copied,
    each later one turns every attribute that differs into '?'. Negative examples are ignored.
    """
    hypothesis = None
    changes = []
    for index, (example, is_positive) in enumerate(zip(examples, positive, strict=True)):
        if not is_positive:
            continue
        if hypothesis is None:
            new = tuple(example)
        else:
            new = tuple(h if h == e else "?" for h, e in zip(hypothesis, example, strict=True))
        if new != hypothesis:
            hypothesis = new
            changes.append((index, hypothesis))
    return hypothesis, changes


def matches(hypothesis: tuple[str, ...] | None, example: tuple[str, ...]) -> bool:
    """True if every attribute of the example equals the hypothesis or the hypothesis says '?'. None fits nothing."""
    if hypothesis is None:
        return False
    return all(h == "?" or h == e for h, e in zip(hypothesis, example, strict=True))
