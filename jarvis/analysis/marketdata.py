"""OHLCV candle fetching from Binance, Yahoo Finance, or a clearly-labelled synthetic walk."""

from __future__ import annotations

import hashlib
import math
import random
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from jarvis._http import httpx
from jarvis.analysis.indicators import Candle
from jarvis.skills.base import SkillContext, SkillError
from jarvis.skills.util import http_client

INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
SYNTHETIC_SOURCE = "synthetic (not real market data)"
_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
_CRYPTO_SUFFIXES = ("USDT", "USDC", "BUSD", "BTC", "ETH")
_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
            "1w": 604800}

_YAHOO_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h",
                   "1d": "1d", "1w": "1wk"}
_YAHOO_RANGES = [("1d", 1), ("5d", 5), ("7d", 7), ("1mo", 30), ("60d", 60), ("3mo", 90),
                 ("6mo", 180), ("1y", 365), ("2y", 730), ("5y", 1825), ("10y", 3650),
                 ("max", 36500)]
_YAHOO_MAX_DAYS = {"1m": 7, "5m": 60, "15m": 60, "1h": 730}
_BARS_PER_DAY = {"1m": 390, "5m": 78, "15m": 26, "1h": 7}


def normalize_symbol(symbol: str) -> str:
    cleaned = (symbol or "").strip().upper()
    if not _SYMBOL_RE.match(cleaned):
        raise SkillError(
            f"Invalid symbol '{symbol}'. Use up to 15 letters, digits or . - ^ = (e.g. AAPL, BTCUSDT)."
        )
    return cleaned


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_crypto_pair(symbol: str) -> bool:
    if any(ch in symbol for ch in ".-^"):
        return False
    return any(symbol.endswith(s) and len(symbol) > len(s) for s in _CRYPTO_SUFFIXES)


async def fetch_candles(
    ctx: SkillContext, symbol: str, interval: str = "1d", limit: int = 200
) -> tuple[list[Candle], str]:
    """Return (candles oldest first, source name). Never substitutes fake data for a failure."""
    if interval not in INTERVALS:
        raise SkillError(f"Invalid interval '{interval}'. Valid intervals: {', '.join(INTERVALS)}.")
    symbol = normalize_symbol(symbol)
    limit = max(1, min(int(limit), 1000))
    source = str(ctx.config.get("trading.data_source", "auto")).lower()
    if source == "auto":
        source = "binance" if _is_crypto_pair(symbol) else "yahoo"
    if source == "synthetic":
        return _synthetic(symbol, interval, limit), SYNTHETIC_SOURCE
    if source == "binance":
        return await _binance(ctx, symbol, interval, limit), "binance"
    if source == "yahoo":
        return await _yahoo(ctx, symbol, interval, limit), "yahoo"
    raise SkillError(
        f"Unknown trading.data_source '{source}'. Use auto, binance, yahoo or synthetic."
    )


async def _get_json(
    ctx: SkillContext,
    name: str,
    url: str,
    params: dict[str, Any],
    symbol: str,
    headers: dict[str, str] | None = None,
):
    try:
        async with http_client(ctx, headers=headers or {}) as client:
            response = await client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise SkillError(f"Could not reach {name} market data: {exc or type(exc).__name__}.") from exc
    if response.status_code in (400, 404):
        raise SkillError(f"Unknown symbol '{symbol}' on {name}.")
    if response.status_code >= 400:
        raise SkillError(f"{name} returned HTTP {response.status_code}.")
    try:
        return response.json()
    except ValueError as exc:
        raise SkillError(f"{name} returned an unreadable response.") from exc


def _binance_base(ctx: SkillContext) -> str:
    """Binance's public market-data host; needs no account and is reachable from
    regions where api.binance.com is not. Override with trading.binance_url."""
    return str(ctx.config.get("trading.binance_url") or "https://data-api.binance.vision").rstrip("/")


async def _binance(ctx: SkillContext, symbol: str, interval: str, limit: int) -> list[Candle]:
    data = await _get_json(
        ctx, "binance", f"{_binance_base(ctx)}/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}, symbol,
    )
    try:
        candles = [
            Candle(_iso(row[0] / 1000), float(row[1]), float(row[2]), float(row[3]),
                   float(row[4]), float(row[5]))
            for row in data
        ]
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise SkillError("binance returned data in an unexpected format.") from exc
    if not candles:
        raise SkillError(f"Unknown symbol '{symbol}' on binance (no candles returned).")
    return candles


def _yahoo_range(interval: str, limit: int) -> str:
    bars = limit * 4 if interval == "4h" else limit
    if interval in _BARS_PER_DAY:
        days = bars / _BARS_PER_DAY[interval] * 1.5 + 1
    elif interval == "1d":
        days = bars * 7 / 5 + 3
    else:
        days = bars * 7 + 7
    cap = _YAHOO_MAX_DAYS.get(_YAHOO_INTERVAL[interval])
    for name, span in _YAHOO_RANGES:
        if span >= days or (cap is not None and span >= cap):
            return name
    return "max"


async def _yahoo(ctx: SkillContext, symbol: str, interval: str, limit: int) -> list[Candle]:
    data = await _get_json(
        ctx, "yahoo", f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        {"interval": _YAHOO_INTERVAL[interval], "range": _yahoo_range(interval, limit)}, symbol,
        # Yahoo turns away clients that do not look like a browser.
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/128.0"},
    )
    chart = data.get("chart") if isinstance(data, dict) else None
    if not isinstance(chart, dict) or chart.get("error") or not chart.get("result"):
        raise SkillError(f"Unknown symbol '{symbol}' on yahoo.")
    try:
        result = chart["result"][0]
        stamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        columns = [quote[k] for k in ("open", "high", "low", "close", "volume")]
        candles = []
        for i, stamp in enumerate(stamps):
            values = [col[i] for col in columns]
            if None in values[:4]:
                continue
            o, h, low, c = (float(v) for v in values[:4])
            candles.append(Candle(_iso(stamp), o, h, low, c, float(values[4] or 0.0)))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SkillError("yahoo returned data in an unexpected format.") from exc
    if not candles:
        raise SkillError(f"Unknown symbol '{symbol}' on yahoo (no price data).")
    if interval == "4h":
        candles = _aggregate(candles, 4)
    return candles[-limit:]


def _aggregate(candles: list[Candle], size: int) -> list[Candle]:
    """Merge consecutive bars in groups of ``size``, aligned so the newest group is full."""
    start = len(candles) % size
    out = []
    for i in range(start, len(candles), size):
        group = candles[i : i + size]
        out.append(Candle(group[0].time, group[0].open, max(c.high for c in group),
                          min(c.low for c in group), group[-1].close,
                          sum(c.volume for c in group)))
    return out


def _synthetic(symbol: str, interval: str, limit: int) -> list[Candle]:
    digest = hashlib.sha256(f"{symbol}|{interval}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    price = rng.uniform(20, 500)
    step = _SECONDS[interval]
    vol = 0.02 * math.sqrt(min(step, 86400) / 86400) + 0.002
    start = datetime(2024, 1, 1, tzinfo=UTC) - timedelta(seconds=step * limit)
    candles = []
    for i in range(limit):
        open_ = price
        close = open_ * math.exp(rng.gauss(0.0002, vol))
        high = max(open_, close) * (1 + abs(rng.gauss(0, vol / 2)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, vol / 2)))
        volume = rng.uniform(0.5, 1.5) * 1_000_000
        candles.append(Candle(_iso(start.timestamp() + step * i), open_, high, low, close, volume))
        price = close
    return candles
