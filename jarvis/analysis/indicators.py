"""Pure-Python technical indicators and a rule-based summary of a price series."""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_CANDLES = 30


@dataclass
class Candle:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period < 1:
        return out
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def _ema_optional(values: list[float | None], period: int) -> list[float | None]:
    """EMA over a series that may start with Nones, seeded with an SMA."""
    out: list[float | None] = [None] * len(values)
    first = next((i for i, v in enumerate(values) if v is not None), None)
    if first is None or period < 1:
        return out
    seed_end = first + period - 1
    if seed_end >= len(values):
        return out
    window = [v for v in values[first : seed_end + 1] if v is not None]
    prev = sum(window) / period
    out[seed_end] = prev
    alpha = 2.0 / (period + 1)
    for i in range(seed_end + 1, len(values)):
        value = values[i]
        if value is None:
            continue
        prev = alpha * value + (1 - alpha) * prev
        out[i] = prev
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    return _ema_optional(list(values), period)


def rsi(values: list[float], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) <= period or period < 1:
        return out
    gains = [max(values[i] - values[i - 1], 0.0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0.0) for i in range(1, len(values))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def value(gain: float, loss: float) -> float:
        if loss == 0:
            return 50.0 if gain == 0 else 100.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    out[period] = value(avg_gain, avg_loss)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = value(avg_gain, avg_loss)
    return out


def macd(
    values: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    fast_line, slow_line = ema(values, fast), ema(values, slow)
    line: list[float | None] = [
        f - s if f is not None and s is not None else None
        for f, s in zip(fast_line, slow_line, strict=True)
    ]
    signal_line = _ema_optional(line, signal)
    hist: list[float | None] = [
        m - s if m is not None and s is not None else None
        for m, s in zip(line, signal_line, strict=True)
    ]
    return line, signal_line, hist


def bollinger(
    values: list[float], period: int = 20, k: float = 2.0
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    middle = sma(values, period)
    lower: list[float | None] = [None] * len(values)
    upper: list[float | None] = [None] * len(values)
    for i, mid in enumerate(middle):
        if mid is None:
            continue
        window = values[i - period + 1 : i + 1]
        std = math.sqrt(sum((v - mid) ** 2 for v in window) / period)
        lower[i], upper[i] = mid - k * std, mid + k * std
    return lower, middle, upper


def atr(candles: list[Candle], period: int = 14) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if len(candles) < period or period < 1:
        return out
    ranges = [candles[0].high - candles[0].low]
    for prev, cur in zip(candles, candles[1:], strict=False):
        ranges.append(
            max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        )
    prev_atr = sum(ranges[:period]) / period
    out[period - 1] = prev_atr
    for i in range(period, len(candles)):
        prev_atr = (prev_atr * (period - 1) + ranges[i]) / period
        out[i] = prev_atr
    return out


def _last(series: list[float | None]) -> float | None:
    return series[-1] if series else None


def _at(series: list[float | None], index: int) -> float | None:
    return series[index] if -len(series) <= index < len(series) else None


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _trend(closes: list[float], sma20, sma50) -> str:
    price = closes[-1]
    votes: list[int] = []
    for line in (sma20, sma50):
        level = _last(line)
        if level is not None:
            votes.append(1 if price > level else -1)
    fast, fast_before = _last(sma20), _at(sma20, -6)
    if fast is not None and fast_before is not None:
        votes.append(1 if fast > fast_before else -1)
    if _last(sma50) is not None and fast is not None:
        votes.append(1 if fast > _last(sma50) else -1)
    total = sum(votes)
    if total >= len(votes) - 1 and total >= 2:
        return "up"
    if total <= -(len(votes) - 1) and total <= -2:
        return "down"
    return "sideways"


def _signals(
    closes: list[float], info: dict, lines: dict[str, list[float | None]]
) -> list[tuple[str, int]]:
    """Each signal is (description, points); positive points are bullish."""
    out: list[tuple[str, int]] = []
    price = closes[-1]
    trend = info["trend"]
    if trend == "up":
        out.append(("Trend is up (price above rising moving averages)", 25))
    elif trend == "down":
        out.append(("Trend is down (price below falling moving averages)", -25))

    for name in ("sma20", "sma50", "sma200"):
        level = info.get(name)
        if level is not None:
            above = price > level
            word = "above" if above else "below"
            out.append((f"Price {word} {name.upper()}", 5 if above else -5))

    rsi_now = info["rsi14"]
    if rsi_now is not None:
        if rsi_now < 30:
            out.append((f"RSI {rsi_now:.1f}: oversold", 10))
        elif rsi_now > 70:
            out.append((f"RSI {rsi_now:.1f}: overbought", -10))

    line, sig = lines["macd"], lines["signal"]
    if _last(sig) is not None and _last(line) is not None:
        if line[-1] > sig[-1]:
            out.append(("MACD above signal line", 10))
        else:
            out.append(("MACD below signal line", -10))
        prev_line, prev_sig = _at(line, -2), _at(sig, -2)
        if prev_line is not None and prev_sig is not None:
            if prev_line <= prev_sig and line[-1] > sig[-1]:
                out.append(("MACD crossed above signal", 15))
            elif prev_line >= prev_sig and line[-1] < sig[-1]:
                out.append(("MACD crossed below signal", -15))

    bands = info["bollinger"]
    if bands["lower"] is not None:
        if price < bands["lower"]:
            out.append(("Price below lower Bollinger band", 10))
        elif price > bands["upper"]:
            out.append(("Price above upper Bollinger band", -10))

    if info["resistance"] is not None and price > info["resistance"]:
        out.append(("Price broke above recent resistance", 15))
    elif info["support"] is not None and price < info["support"]:
        out.append(("Price broke below recent support", -15))

    ratio = info["volume_vs_avg"]
    if ratio is not None and ratio >= 1.5:
        move = info["change_pct"] or 0
        if move > 0:
            out.append((f"Volume {ratio:.1f}x average on an up move", 5))
        elif move < 0:
            out.append((f"Volume {ratio:.1f}x average on a down move", -5))
    return out


def _compute(candles: list[Candle]) -> tuple[dict, list[tuple[str, int]]]:
    if len(candles) < MIN_CANDLES:
        raise ValueError(f"Need at least {MIN_CANDLES} candles, got {len(candles)}.")
    closes = [c.close for c in candles]
    price = closes[-1]
    prev_close = closes[-2]
    change = (price - prev_close) / prev_close * 100 if prev_close else 0.0

    sma20, sma50, sma200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    ema12, ema26 = ema(closes, 12), ema(closes, 26)
    rsi14 = rsi(closes, 14)
    line, sig, hist = macd(closes)
    lower, middle, upper = bollinger(closes)
    atr14 = _last(atr(candles, 14))

    prior = candles[-21:-1]
    support = min(c.low for c in prior)
    resistance = max(c.high for c in prior)

    volumes = [c.volume for c in candles]
    avg_volume = sum(volumes[-21:-1]) / len(volumes[-21:-1])
    volume_ratio = volumes[-1] / avg_volume if avg_volume else None

    info: dict = {
        "last_close": _r(price),
        "change_pct": _r(change, 2),
        "sma20": _r(_last(sma20)),
        "sma50": _r(_last(sma50)),
        "sma200": _r(_last(sma200)),
        "ema12": _r(_last(ema12)),
        "ema26": _r(_last(ema26)),
        "rsi14": _r(_last(rsi14), 1),
        "macd": {"line": _r(_last(line)), "signal": _r(_last(sig)), "hist": _r(_last(hist))},
        "bollinger": {"lower": _r(_last(lower)), "middle": _r(_last(middle)),
                      "upper": _r(_last(upper))},
        "atr14": _r(atr14),
        "trend": _trend(closes, sma20, sma50),
        "support": _r(support),
        "resistance": _r(resistance),
        "volume_vs_avg": _r(volume_ratio, 2),
    }
    signals = _signals(closes, info, {"macd": line, "signal": sig})
    return info, signals


def _bias(score: int) -> str:
    if score >= 20:
        return "bullish"
    if score <= -20:
        return "bearish"
    return "neutral"


def analyze(candles: list[Candle]) -> dict:
    info, signals = _compute(candles)
    score = max(-100, min(100, sum(points for _, points in signals)))
    info["price"] = info.pop("last_close")
    info["signals"] = [text for text, _ in signals]
    info["bias"] = _bias(score)
    info["score"] = score
    return info


def plan_setup(candles: list[Candle], direction: str) -> dict:
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'.")
    info, signals = _compute(candles)
    atr_now = info["atr14"]
    if not atr_now:
        raise ValueError("Not enough price movement to compute ATR.")
    sign = 1 if direction == "long" else -1
    entry = candles[-1].close
    distance = 1.5 * atr_now
    stop = entry - sign * distance
    target = entry + sign * 2 * distance

    favour = [text for text, points in signals if points * sign > 0]
    against = [text for text, points in signals if points * sign < 0]
    directional = sign * max(-100, min(100, sum(points for _, points in signals)))
    verdict = "favourable" if directional >= 20 else "unfavourable" if directional <= -20 else "mixed"
    return {
        "entry": _r(entry),
        "stop": _r(stop),
        "target": _r(target),
        "risk_per_unit": _r(distance),
        "risk_reward": 2.0,
        "atr14": info["atr14"],
        "confluence_for": favour,
        "confluence_against": against,
        "verdict": verdict,
    }
