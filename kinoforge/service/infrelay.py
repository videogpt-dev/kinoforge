from __future__ import annotations

import base64
from typing import Any, Dict, List, Tuple

import httpx


class InfrelayError(RuntimeError):
    pass


class InfrelayClient:
    def __init__(self, url: str, token: str = "", tenant: str = "") -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._tenant = tenant

    def _generate(
        self,
        kind: str,
        provider: str,
        model: str,
        payload: Dict[str, Any],
        *,
        timeout: float,
    ) -> Dict[str, Any]:
        if not self._url:
            raise InfrelayError("INFRELAY_URL is required")
        body: Dict[str, Any] = {
            "kind": kind,
            "provider": provider,
            "model": model,
            "input": payload,
        }
        if self._tenant:
            body["tenant_id"] = self._tenant
        headers = (
            {"Authorization": f"Bearer {self._token}"} if self._token else {}
        )
        try:
            response = httpx.post(
                f"{self._url}/v1/generate",
                headers=headers,
                json=body,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise InfrelayError(f"infrelay {kind}/{provider} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise InfrelayError(
                f"infrelay {kind}/{provider} failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        return response.json()

    def complete(
        self,
        provider: str,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        result = self._generate(
            "text",
            provider,
            model,
            {
                "system": "",
                "user": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        output = result.get("output") or {}
        if output.get("type") != "text":
            raise InfrelayError("infrelay text returned unsupported output")
        return str(output.get("value") or "")

    def text(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, Dict]:
        """One chat completion. Returns (text, usage) — usage carries model, token counts
        and finish_reason so the story stage runner can detect a cut-off answer."""
        result = self._generate(
            "text",
            provider,
            model,
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        output = result.get("output") or {}
        if output.get("type") != "text":
            raise InfrelayError("infrelay text returned unsupported output")
        return str(output.get("value") or ""), dict(output.get("meta") or {})

    def image(
        self,
        prompt: str,
        provider: str,
        model: str,
        *,
        aspect_ratio: str = "9:16",
        seed: int | None = None,
        mature: bool = False,
        negative: str = "",
        reference: bytes | None = None,
    ) -> bytes:
        """Image bytes from the gateway. Shapes the same input the in-process path builds."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "safe": not mature,
        }
        if model:
            payload["model"] = model
        if seed is not None:
            payload["seed"] = int(seed)
        if negative:
            payload["negative"] = negative
        if reference is not None:
            payload["reference_b64"] = base64.b64encode(reference).decode()
        result = self._generate("image", provider, model, payload, timeout=600)
        return self._output_bytes(result.get("output") or {})

    def video(
        self,
        prompt: str,
        provider: str,
        model: str,
        *,
        seconds: float,
        resolution: int,
        aspect_ratio: str,
        mature: bool = False,
        image: bytes | None = None,
    ) -> bytes:
        """Video (mp4) bytes from the gateway."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "seconds": seconds,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "safe": not mature,
        }
        if model:
            payload["model"] = model
        if image is not None:
            payload["reference_b64"] = base64.b64encode(image).decode()
        result = self._generate("video", provider, model, payload, timeout=1800)
        return self._output_bytes(result.get("output") or {})

    def music(self, prompt: str, provider: str, model: str, *, seconds: int = 30) -> bytes:
        """Music (audio) bytes from the gateway."""
        payload: Dict[str, Any] = {"prompt": prompt, "seconds": seconds}
        if model:
            payload["model"] = model
        result = self._generate("music", provider, model, payload, timeout=1800)
        return self._output_bytes(result.get("output") or {})

    def speech(
        self,
        text: str,
        provider: str,
        model: str,
        *,
        voice: str = "",
        language: str = "",
    ) -> bytes:
        payload: Dict[str, Any] = {"text": text}
        if voice:
            payload["voice"] = voice
        if language:
            payload["language"] = language
        result = self._generate("audio", provider, model, payload, timeout=600)
        return self._output_bytes(result.get("output") or {})

    def _output_bytes(self, output: Dict[str, Any]) -> bytes:
        kind = output.get("type")
        if kind == "b64":
            return base64.b64decode(output["value"])
        if kind == "url":
            try:
                response = httpx.get(str(output["value"]), timeout=600, follow_redirects=True)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise InfrelayError(f"could not fetch generated media: {exc}") from exc
            return response.content
        raise InfrelayError(f"infrelay returned unsupported output type: {kind}")

    def transcribe(
        self,
        audio: bytes,
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> Tuple[List[Dict], Dict]:
        payload: Dict[str, Any] = {
            "audio_b64": base64.b64encode(audio).decode(),
        }
        language = kwargs.get("language")
        if language:
            payload["language"] = language
        if kwargs.get("word_timestamps"):
            payload["word_timestamps"] = True
        params = kwargs.get("params") or {}
        payload.update({key: value for key, value in params.items() if value is not None})
        result = self._generate(
            "transcribe",
            provider,
            model,
            payload,
            timeout=1800,
        )
        output = result.get("output") or {}
        if output.get("type") != "transcript":
            raise InfrelayError("infrelay transcribe returned unsupported output")
        meta = output.get("meta") or {}
        return list(meta.get("segments") or []), meta
