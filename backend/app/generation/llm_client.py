"""Multi-provider LLM client with automatic fallback and streaming."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_llm_client: Optional["LLMClient"] = None


class LLMClient:
    """Unified LLM client supporting Groq and Gemini with fallback."""

    def __init__(self):
        self.providers = self._build_provider_chain()

    def _build_provider_chain(self) -> list[dict]:
        """Build ordered list of (provider, api_key, model, base_url)."""
        chain = []

        # Primary
        primary = settings.LLM_PROVIDER.lower()
        if primary == "groq" and settings.GROQ_API_KEY:
            chain.append({
                "name": "groq",
                "api_key": settings.GROQ_API_KEY,
                "model": settings.LLM_MODEL,
                "base_url": "https://api.groq.com/openai/v1",
            })
        elif primary == "gemini" and settings.GEMINI_API_KEY:
            chain.append({
                "name": "gemini",
                "api_key": settings.GEMINI_API_KEY,
                "model": settings.LLM_MODEL,
            })

        # Fallback
        fallback = settings.LLM_FALLBACK_PROVIDER.lower()
        if fallback == "gemini" and settings.GEMINI_API_KEY:
            chain.append({
                "name": "gemini",
                "api_key": settings.GEMINI_API_KEY,
                "model": settings.LLM_FALLBACK_MODEL,
            })
        elif fallback == "groq" and settings.GROQ_API_KEY:
            chain.append({
                "name": "groq",
                "api_key": settings.GROQ_API_KEY,
                "model": settings.LLM_FALLBACK_MODEL,
                "base_url": "https://api.groq.com/openai/v1",
            })

        return chain

    @staticmethod
    def _clean_cot(text: str) -> str:
        """Remove chain-of-thought reasoning from model output.

        Handles:
        - <think>...</think> XML tags
        - Plain-text reasoning preambles from models like openai/gpt-oss-20b
        - Instruction echoes ("Provide citation", "Use only context", "So final")
        """
        import re

        clean = text or ""

        # 1. Remove <think>...</think> blocks
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)

        # 2. Transition markers — keep text AFTER the last match
        cot_transition_patterns = [
            r"(?:Provide final answer|Provide answer)\s*[.:\-]*\s*",
            r"(?:Final [Aa]nswer|FINAL ANSWER)\s*[:\-]*\s*",
            r"(?:So,?\s+(?:the\s+)?answer\s+is|So\s+answer)\s*[:\-]*\s*",
            r"(?:Here is (?:the|my) (?:final )?answer)\s*[:\-]*\s*",
            r"(?:Use inline citations)\s*[.:\-]*\s*",
            r"(?:Provide citations?)\s*(?:\[\d+\](?:\s*or\s*\[\d+\])?)?\s*[.:\-]*\s*",
            r"(?:Use only context)\s*[.:\-]*\s*",
            r"(?:So\s+final)\s*[.:\-]*\s*",
        ]

        best_end = -1
        for pattern in cot_transition_patterns:
            for match in re.finditer(pattern, clean, re.IGNORECASE):
                if match.end() > best_end:
                    best_end = match.end()

        if best_end > 0:
            candidate = clean[best_end:].strip()
            if len(candidate) >= 8:
                clean = candidate

        # 3. Strip leftover instruction-echo spans if still present
        noise = re.compile(
            r"(?:provide citations?(?:\s*\[\d+\](?:\s*or\s*\[\d+\])?)?"
            r"|use only context"
            r"|so final"
            r"|use inline citations)"
            r"\s*[.:\-]*",
            re.IGNORECASE,
        )
        if noise.search(clean):
            parts = noise.split(clean)
            tail = parts[-1].strip() if parts else ""
            if len(tail) >= 8:
                clean = tail

        # 4. Prefer the last cited answer sentence if multiple drafts remain
        cited = list(
            re.finditer(
                r"((?:[A-Z\"][^.!?\n]*|[^.!?\n]{8,})[.!]?\s*\[\d+\])",
                clean,
            )
        )
        if cited:
            clean = cited[-1].group(1).strip()

        # 5. Unwrap quotes / tidy
        clean = clean.strip().strip('"').strip("'").strip()
        clean = re.sub(r"\s+", " ", clean)

        # 6. Safety fallback
        if len(clean) < 5:
            fallback = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
            return fallback if fallback else (text or "").strip()

        return clean

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion, falling through providers on failure."""
        for provider in self.providers:
            try:
                if provider["name"] == "groq":
                    return await self._groq_generate(provider, prompt, system, max_tokens)
                elif provider["name"] == "gemini":
                    return await self._gemini_generate(provider, prompt, system, max_tokens)
            except Exception as exc:
                logger.warning(
                    "LLM provider %s/%s failed: %s",
                    provider["name"], provider["model"], exc,
                )
                continue

        raise RuntimeError("All LLM providers failed")

    async def generate_stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Stream tokens from the first available provider."""
        for provider in self.providers:
            try:
                logger.info("[STREAM] Trying provider %s/%s", provider["name"], provider["model"])
                if provider["name"] == "groq":
                    gen = self._groq_stream(provider, prompt, system, max_tokens)
                elif provider["name"] == "gemini":
                    gen = self._gemini_stream(provider, prompt, system, max_tokens)
                else:
                    continue

                # Buffer the entire response to clean CoT reasoning
                buffer = ""
                async for token in gen:
                    buffer += token

                logger.info("[STREAM] Raw buffer length: %d, content: '%s'", len(buffer), buffer[:200])
                
                # Clean chain-of-thought
                clean_text = self._clean_cot(buffer)
                logger.info("[STREAM] Clean text length: %d, content: '%s'", len(clean_text), clean_text[:200])

                # Yield the cleaned text in chunks to simulate streaming for the UI
                chunk_size = 4
                for i in range(0, len(clean_text), chunk_size):
                    yield clean_text[i:i+chunk_size]

                return
            except Exception as exc:
                logger.warning(
                    "Streaming %s/%s failed: %s",
                    provider["name"], provider["model"], exc,
                )
                continue
        
        logger.error("[STREAM] All providers failed, no tokens generated")

    # ── Groq (OpenAI-compatible) ───────────────────────

    async def _groq_generate(
        self, provider: dict, prompt: str, system: str, max_tokens: int
    ) -> str:
        import asyncio as _asyncio
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        max_retries = 3
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{provider['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {provider['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": provider["model"],
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                    },
                )
                if resp.status_code == 429 and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 1  # 1s, 2s, 4s
                    logger.warning("Groq rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    await _asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content", "")
                if not content:
                    content = msg.get("reasoning_content", "") or msg.get("reasoning", "")
                return self._clean_cot(content)

    async def _groq_stream(
        self, provider: dict, prompt: str, system: str, max_tokens: int
    ) -> AsyncIterator[str]:
        import asyncio as _asyncio
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        max_retries = 3
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{provider['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {provider['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": provider["model"],
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                        "stream": True,
                    },
                ) as resp:
                    if resp.status_code == 429 and attempt < max_retries - 1:
                        wait = (2 ** attempt) * 1
                        logger.warning("Groq stream rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                        await _asyncio.sleep(wait)
                        break  # Break inner 'async with' to retry
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                                
                            # Yield standard content tokens
                            content = delta.get("content", "")
                            if content:
                                yield content
                            
                            # Also capture reasoning_content for reasoning models
                            reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                            if reasoning and not content:
                                yield reasoning
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                    
                    # Successfully completed the stream, exit the retry loop
                    return

    # ── Gemini ─────────────────────────────────────────

    async def _gemini_generate(
        self, provider: dict, prompt: str, system: str, max_tokens: int
    ) -> str:
        import asyncio as _asyncio
        
        model = provider["model"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={provider['api_key']}"
        )
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        max_retries = 3
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=body)
                if resp.status_code == 429 and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 1
                    logger.warning("Gemini rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                    await _asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return self._clean_cot(
                    data["candidates"][0]["content"]["parts"][0]["text"]
                )
        
        raise RuntimeError("Gemini generate failed after retries")

    async def _gemini_stream(
        self, provider: dict, prompt: str, system: str, max_tokens: int
    ) -> AsyncIterator[str]:
        import asyncio as _asyncio
        model = provider["model"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":streamGenerateContent?alt=sse&key={provider['api_key']}"
        )
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        max_retries = 3
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=body) as resp:
                    if resp.status_code == 429 and attempt < max_retries - 1:
                        wait = (2 ** attempt) * 1
                        logger.warning("Gemini stream rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                        await _asyncio.sleep(wait)
                        break  # Break inner 'async with' to retry
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                            parts = (
                                data.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [])
                            )
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    yield text
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                    
                    # Successfully completed the stream, exit the retry loop
                    return


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
