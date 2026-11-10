"""Single import point for the HTTP client (the Anthropic SDK uses httpx2 too)."""

import httpx2 as httpx

__all__ = ["httpx"]
