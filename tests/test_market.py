import asyncio
import math

import pytest

from jarvis._http import httpx
from jarvis.analysis.indicators import (
    Candle,
    analyze,
    atr,
    bollinger,
    ema,
    macd,
    plan_setup,
    rsi,
    sma,
)
from jarvis.analysis.marketdata import SYNTHETIC_SOURCE, fetch_candles
from jarvis.skills.base import SkillError
from jarvis.skills.builtin.market import (
    evaluate_setup,
    market_quote,
    market_scan,
    technical_analysis,
)

WILDER = [
    44.34,
    44.09,
    44.15,
    43.61,
    44.33,
    44.83,
    45.10,
    45.42,
    45.84,
    46.08,
    45.89,
    46.03,
    45.61,
    46.28,
    46.28,
    46.00,
    46.03,
    46.41,
    46.22,
    45.64,
]


def make_candles(closes, spread=0.5, volume=1000.0):
    return [Candle(f"t{i}", c, c + spread, c - spread, c, volume) for i, c in enumerate(closes)]


def rising(n=120):
    return make_candles([100 + i * 0.5 + math.sin(i) * 0.3 for i in range(n)])


def falling(n=120):
    return make_candles([200 - i * 0.5 + math.sin(i) * 0.3 for i in range(n)])


def test_sma():
    assert sma([1, 2, 3, 4, 5], 3) == [None, None, 2.0, 3.0, 4.0]


def test_ema_seeded_with_sma():
    out = ema([1, 2, 3, 4, 5], 3)
    assert out[:2] == [None, None]
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_rsi_wilder_example():
    out = rsi(WILDER, 14)
    assert out[13] is None
    assert out[14] == pytest.approx(70.53, abs=0.1)
    assert out[15] == pytest.approx(66.32, abs=0.1)


def test_rsi_monotonic():
    assert rsi([float(i) for i in range(1, 30)])[-1] == 100.0
    assert rsi([float(i) for i in range(30, 1, -1)])[-1] == 0.0


def test_macd_rising_positive():
    line, signal, hist = macd([float(i) for i in range(1, 80)])
    assert line[-1] > 0
    assert len(line) == len(signal) == len(hist) == 79
    assert line[24] is None and line[25] is not None
    assert signal[-1] is not None


def test_bollinger_middle_is_sma():
    values = [math.sin(i) * 3 + 50 for i in range(40)]
    lower, middle, upper = bollinger(values, 20)
    assert middle == sma(values, 20)
    assert lower[-1] < middle[-1] < upper[-1]
    assert lower[18] is None


def test_atr_constant_range():
    candles = make_candles([100.0] * 30, spread=1.0)
    assert atr(candles, 14)[-1] == pytest.approx(2.0)


def test_analyze_rising_vs_falling():
    up, down = analyze(rising()), analyze(falling())
    assert up["bias"] == "bullish" and up["trend"] == "up" and up["score"] > 0
    assert down["bias"] == "bearish" and down["trend"] == "down" and down["score"] < 0
    assert -100 <= up["score"] <= 100
    assert up["sma200"] is None and up["sma50"] is not None


def test_analyze_needs_30_candles():
    with pytest.raises(ValueError):
        analyze(rising(29))


def test_plan_setup():
    candles = rising()
    long = plan_setup(candles, "long")
    short = plan_setup(candles, "short")
    assert long["stop"] < long["entry"] < long["target"]
    assert short["target"] < short["entry"] < short["stop"]
    assert long["risk_reward"] == 2.0
    assert (long["target"] - long["entry"]) == pytest.approx(
        2 * (long["entry"] - long["stop"]), abs=1e-3
    )
    assert long["verdict"] == "favourable"
    assert short["verdict"] == "unfavourable"
    assert long["confluence_for"] == short["confluence_against"]


def test_synthetic_deterministic(ctx):
    a, source = asyncio.run(fetch_candles(ctx, "aapl ", "1d", 60))
    b, _ = asyncio.run(fetch_candles(ctx, "AAPL", "1d", 60))
    c, _ = asyncio.run(fetch_candles(ctx, "MSFT", "1d", 60))
    assert source == SYNTHETIC_SOURCE
    assert a == b and len(a) == 60
    assert a != c
    assert all(x.low <= min(x.open, x.close) and x.high >= max(x.open, x.close) for x in a)


def test_invalid_interval_and_symbol(ctx):
    with pytest.raises(SkillError, match="1m.*1w"):
        asyncio.run(fetch_candles(ctx, "AAPL", "2d"))
    for bad in ("", "A B", "AAPL;DROP", "TOOLONGSYMBOL12345"):
        with pytest.raises(SkillError):
            asyncio.run(fetch_candles(ctx, bad, "1d"))


def use_transport(ctx, handler):
    ctx.config.set("trading.data_source", "auto")
    ctx.services["http_transport"] = httpx.MockTransport(handler)


def test_binance_parsing(ctx):
    seen = []

    def handler(request):
        seen.append(request)
        rows = [[1700000000000 + i * 86400000, "10", "12", "9", "11", "5.5", 0] for i in range(3)]
        return httpx.Response(200, json=rows)

    use_transport(ctx, handler)
    candles, source = asyncio.run(fetch_candles(ctx, "btcusdt", "1d", 3))
    assert source == "binance"
    assert seen[0].url.host == "data-api.binance.vision"
    assert seen[0].url.params["symbol"] == "BTCUSDT"
    assert candles[0] == Candle("2023-11-14T22:13:20Z", 10.0, 12.0, 9.0, 11.0, 5.5)


def yahoo_payload(n=4, skip=None):
    stamps = [1700000000 + i * 3600 for i in range(n)]
    close = [10.0 + i for i in range(n)]
    if skip is not None:
        close[skip] = None
    quote = {
        "open": [c if c is None else c - 0.5 for c in close],
        "high": [c if c is None else c + 1 for c in close],
        "low": [c if c is None else c - 1 for c in close],
        "close": close,
        "volume": [100] * n,
    }
    return {
        "chart": {
            "result": [{"timestamp": stamps, "indicators": {"quote": [quote]}}],
            "error": None,
        }
    }


def test_yahoo_parsing_skips_null_bars(ctx):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=yahoo_payload(4, skip=1))

    use_transport(ctx, handler)
    candles, source = asyncio.run(fetch_candles(ctx, "AAPL", "1d", 10))
    assert source == "yahoo"
    assert seen[0].url.host == "query1.finance.yahoo.com"
    assert seen[0].url.path.endswith("/AAPL")
    assert [c.close for c in candles] == [10.0, 12.0, 13.0]


def test_yahoo_4h_aggregation_and_weekly_interval(ctx):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=yahoo_payload(8))

    use_transport(ctx, handler)
    candles, _ = asyncio.run(fetch_candles(ctx, "AAPL", "4h", 10))
    assert seen[0].url.params["interval"] == "1h"
    assert len(candles) == 2
    assert candles[0].open == 9.5 and candles[0].close == 13.0 and candles[0].volume == 400.0
    assert candles[0].high == 14.0 and candles[0].low == 9.0
    asyncio.run(fetch_candles(ctx, "AAPL", "1w", 10))
    assert seen[-1].url.params["interval"] == "1wk"


def test_yahoo_unknown_symbol(ctx):
    body = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    use_transport(ctx, lambda request: httpx.Response(200, json=body))
    with pytest.raises(SkillError, match="Unknown symbol"):
        asyncio.run(fetch_candles(ctx, "ZZZZ", "1d"))


def test_auto_routing_dotted_goes_to_yahoo(ctx):
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        return httpx.Response(200, json=yahoo_payload())

    use_transport(ctx, handler)
    asyncio.run(fetch_candles(ctx, "BTC-USD", "1d", 5))
    asyncio.run(fetch_candles(ctx, "USDT", "1d", 5))
    assert hosts == ["query1.finance.yahoo.com"] * 2


def test_network_failure_raises_not_synthetic(ctx):
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    use_transport(ctx, handler)
    with pytest.raises(SkillError, match="binance"):
        asyncio.run(fetch_candles(ctx, "ETHUSDT", "1d"))
    with pytest.raises(SkillError, match="yahoo"):
        asyncio.run(fetch_candles(ctx, "AAPL", "1d"))


def test_http_error_status(ctx):
    use_transport(ctx, lambda request: httpx.Response(503))
    with pytest.raises(SkillError, match="503"):
        asyncio.run(fetch_candles(ctx, "AAPL", "1d"))


def test_skills_with_synthetic_data(ctx):
    quote = asyncio.run(market_quote("aapl", ctx))
    assert quote["symbol"] == "AAPL" and quote["source"] == SYNTHETIC_SOURCE
    assert quote["note"] == "Analysis only, not financial advice."
    ta = asyncio.run(technical_analysis("AAPL", ctx, "1d"))
    assert ta["interval"] == "1d" and ta["bias"] in ("bullish", "bearish", "neutral")
    setup = asyncio.run(evaluate_setup("AAPL", "long", ctx))
    assert setup["direction"] == "long" and setup["risk_reward"] == 2.0
    assert market_quote.__skill__.tags == ["market"]
    assert evaluate_setup.__skill__.parameters["properties"]["direction"]["enum"] == [
        "long",
        "short",
    ]


def test_market_scan_one_failing_symbol(ctx):
    def handler(request):
        if request.url.path.endswith("/BAD"):
            return httpx.Response(404)
        return httpx.Response(200, json=yahoo_payload(60))

    use_transport(ctx, handler)
    result = asyncio.run(market_scan(ctx, symbols=["AAA", "BAD", "CCC"]))
    rows = result["results"]
    assert [r["symbol"] for r in rows][-1] == "BAD"
    assert "error" in rows[-1]
    good = rows[:-1]
    assert len(good) == 2
    assert good[0]["score"] >= good[1]["score"]
    assert set(good[0]) == {"symbol", "price", "change_pct", "rsi", "trend", "bias", "score"}


def test_market_scan_uses_watchlist_and_caps(ctx):
    ctx.config.set("trading.watchlist", [])
    with pytest.raises(SkillError, match="watchlist"):
        asyncio.run(market_scan(ctx))
    ctx.config.set("trading.watchlist", [f"S{i}" for i in range(40)])
    result = asyncio.run(market_scan(ctx))
    assert len(result["results"]) == 25
    assert "truncated" in result
