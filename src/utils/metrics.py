def pct(n: int | float, d: int | float) -> float:
    return round((n / d * 100) if d else 0.0, 4)
