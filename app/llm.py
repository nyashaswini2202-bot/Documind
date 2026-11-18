"""LLM client. Providers: Groq (hosted, OpenAI-compatible API) or Ollama (local), via plain HTTP.

If no provider is configured the app still works in "extractive" mode and returns the top passages.
"""
from __future__ import annotations

import json
import time
from typing import Iterator, Optional

import requests

from .config import settings


class LLMError(RuntimeError):
    """Raised for any provider failure; the message is safe to show to the user."""


class LLMClient:
    def __init__(self) -> None:
        self.provider = settings.resolved_provider

    @property
    def available(self) -> bool:
        return self.provider != "none"

    @property
    def model(self) -> str:
        return {"groq": settings.groq_model, "ollama": settings.ollama_model}.get(self.provider, "")

    # ------------------------------------------------------------------ public API
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        temperature = settings.llm_temperature if temperature is None else temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        if self.provider == "groq":
            return self._groq_chat(messages, temperature, max_tokens, json_mode)
        if self.provider == "ollama":
            return self._ollama_chat(messages, temperature, max_tokens, json_mode)
        raise LLMError("No LLM configured. Set GROQ_API_KEY (or LLM_PROVIDER=ollama) in your .env file.")

    def stream(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        temperature = settings.llm_temperature if temperature is None else temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        if self.provider == "groq":
            yield from self._groq_stream(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            yield from self._ollama_stream(messages, temperature, max_tokens)
        else:
            raise LLMError("No LLM configured.")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _error_text(resp: requests.Response) -> str:
        try:
            body = resp.json()
            err = body.get("error", body)
            msg = err.get("message") if isinstance(err, dict) else str(err)
        except Exception:
            msg = resp.text[:200]
        if resp.status_code == 401:
            return "The LLM API key was rejected. Check GROQ_API_KEY."
        if resp.status_code == 429:
            return "The LLM rate limit was reached. Wait a few seconds and try again."
        return f"LLM request failed ({resp.status_code}): {msg}"

    def _post(self, url: str, *, headers: dict, payload: dict, stream: bool = False) -> requests.Response:
        last: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, stream=stream, timeout=(10, 120))
            except requests.RequestException as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                wait = min(float(resp.headers.get("retry-after", 2 * (attempt + 1))), 10.0)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise LLMError(self._error_text(resp))
            return resp
        raise LLMError(f"Could not reach the LLM service ({last.__class__.__name__ if last else 'unknown error'}).")

    # ------------------------------------------------------------------ Groq
    def _groq_request(self, messages, temperature, max_tokens, stream, json_mode=False):
        print("DEBUG: GROQ MODEL =", settings.groq_model)
        print("DEBUG: GROQ PROVIDER =", settings.resolved_provider)
        payload = {
            "model": settings.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"}
        return self._post(f"{settings.groq_base_url}/chat/completions", headers=headers, payload=payload, stream=stream)

    def _groq_chat(self, messages, temperature, max_tokens, json_mode) -> str:
        resp = self._groq_request(messages, temperature, max_tokens, False, json_mode)
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Unexpected response from the LLM service.") from exc

    def _groq_stream(self, messages, temperature, max_tokens) -> Iterator[str]:
        resp = self._groq_request(messages, temperature, max_tokens, True)
        resp.encoding = "utf-8"
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                yield delta

    # ------------------------------------------------------------------ Ollama
    def _ollama_payload(self, messages, temperature, max_tokens, stream, json_mode=False) -> dict:
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def _ollama_chat(self, messages, temperature, max_tokens, json_mode) -> str:
        payload = self._ollama_payload(messages, temperature, max_tokens, False, json_mode)
        resp = self._post(f"{settings.ollama_url}/api/chat", headers={}, payload=payload)
        try:
            return resp.json()["message"]["content"] or ""
        except (KeyError, ValueError) as exc:
            raise LLMError("Unexpected response from Ollama.") from exc

    def _ollama_stream(self, messages, temperature, max_tokens) -> Iterator[str]:
        payload = self._ollama_payload(messages, temperature, max_tokens, True)
        resp = self._post(f"{settings.ollama_url}/api/chat", headers={}, payload=payload, stream=True)
        resp.encoding = "utf-8"
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            text = chunk.get("message", {}).get("content")
            if text:
                yield text
            if chunk.get("done"):
                break
