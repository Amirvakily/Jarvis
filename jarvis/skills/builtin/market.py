"""Market skills: quotes, technical analysis, watchlist scans and trade-setup evaluation.

These skills analyse only; there is deliberately no order execution.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from jarvis.analysis.indicators import analyze, plan_setup
from jarvis.analysis.marketdata import fetch_candles, normalize_symbol
from jarvis.skills.base import SkillContext, SkillError, skill

NOTE = "Analysis only, not financial advice."
MAX_SCAN = 25


@skill(tags=["market"])
async def market_quote(symbol: str, ctx: SkillContext) -> dict:
    """Get the latest price, daily change and volume for a stock, ETF, index or crypto pair.

    Args:
        symbol: Ticker such as AAPL, SPY, ^GSPC or a crypto pair such as BTCUSDT.
    """
    symbol = normalize_symbol(symbol)
    candles, source = await fetch_candles(ctx, symbol, "1d", 5)
    last = candles[-1]
    change = None
    if len(candles) > 1 and candles[-2].close:
        change = round((last.close - candles[-2].close) / candles[-2].close * 100, 2)
    return {
        "symbol": symbol,
        "price": round(last.close, 4),
        "change_pct": change,
        "day_high": round(last.high, 4),
        "day_low": round(last.low, 4),
        "volume": round(last.volume, 2),
        "source": source,
        "as_of": last.time,
        "note": NOTE,
    }


def _analysis(candles, symbol: str) -> dict:
    try:
        return analyze(candles)
    except ValueError as exc:
        raise SkillError(f"Not enough price history for {symbol}: {exc}") from exc


@skill(tags=["market"])
async def technical_analysis(symbol: str, ctx: SkillContext, interval: str = "1d") -> dict:
    """Run technical analysis (trend, RSI, MACD, Bollinger, ATR, support/resistance) on a symbol.

    Use it to judge whether a market looks bullish, bearish or neutral.

    Args:
        symbol: Ticker such as AAPL or a crypto pair such as BTCUSDT.
        interval: Candle size: 1m, 5m, 15m, 1h, 4h, 1d or 1w.
    """
    symbol = normalize_symbol(symbol)
    candles, source = await fetch_candles(ctx, symbol, interval, 250)
    return {"symbol": symbol, "interval": interval, "source": source,
            **_analysis(candles, symbol), "note": NOTE}


@skill(tags=["market"])
async def market_scan(
    ctx: SkillContext, symbols: list[str] | None = None, interval: str = "1d"
) -> dict:
    """Scan several symbols at once and rank them by technical score, strongest first.

    With no symbols given, the configured watchlist is scanned.

    Args:
        symbols: Tickers to scan (max 25). Omit to use the watchlist.
        interval: Candle size: 1m, 5m, 15m, 1h, 4h, 1d or 1w.
    """
    chosen = symbols or ctx.config.get("trading.watchlist", []) or []
    if isinstance(chosen, str):
        chosen = [chosen]
    chosen = list(dict.fromkeys(str(s).strip().upper() for s in chosen if str(s).strip()))
    if not chosen:
        raise SkillError("No symbols given and trading.watchlist is empty.")
    truncated = len(chosen) > MAX_SCAN
    chosen = chosen[:MAX_SCAN]

    async def scan_one(symbol: str) -> dict:
        try:
            candles, _ = await fetch_candles(ctx, symbol, interval, 250)
            result = _analysis(candles, symbol)
        except SkillError as exc:
            return {"symbol": symbol, "error": str(exc)}
        return {"symbol": symbol, "price": result["price"], "change_pct": result["change_pct"],
                "rsi": result["rsi14"], "trend": result["trend"], "bias": result["bias"],
                "score": result["score"]}

    rows = await asyncio.gather(*(scan_one(s) for s in chosen))
    ok = sorted((r for r in rows if "error" not in r), key=lambda r: r["score"], reverse=True)
    failed = [r for r in rows if "error" in r]
    out: dict = {"interval": interval, "results": ok + failed, "note": NOTE}
    if truncated:
        out["truncated"] = f"Only the first {MAX_SCAN} symbols were scanned."
    return out


@skill(tags=["market"])
async def evaluate_setup(
    symbol: str,
    direction: Literal["long", "short"],
    ctx: SkillContext,
    interval: str = "1d",
) -> dict:
    """Evaluate a hypothetical long or short trade: entry, ATR-based stop, target and confluence.

    Args:
        symbol: Ticker such as AAPL or a crypto pair such as BTCUSDT.
        direction: "long" to bet on a rise, "short" to bet on a fall.
        interval: Candle size: 1m, 5m, 15m, 1h, 4h, 1d or 1w.
    """
    symbol = normalize_symbol(symbol)
    candles, source = await fetch_candles(ctx, symbol, interval, 250)
    try:
        plan = plan_setup(candles, direction)
    except ValueError as exc:
        raise SkillError(f"Cannot evaluate {symbol}: {exc}") from exc
    return {"symbol": symbol, "direction": direction, "source": source, **plan, "note": NOTE}
