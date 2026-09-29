from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Tuple

import httpx

from kinoforge.contract import ImageSpec, ModelRef, VideoSpec
from kinoforge.observ import active, shape


class InfrelayError(RuntimeError):
    pass


class InfrelayClient:
    """Client for the Infrelay /v1/generate contract (cloud Infrelay or Infrelay-lite)."""

    def __init__(self, url: str, token: str = "", tenant: str = "") -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._tenant = tenant

    def _generate(
        self, kind: str, ref: ModelRef, payload: Dict[str, Any], *, timeout: float
    ) -> Dict[str, Any]:
        if not self._url:
            raise InfrelayError("INFRELAY_URL is required")
        body: Dict[str, Any] = {
            "kind": kind, "provider": ref.provider, "model": ref.model, "input": payload,
        }
        if self._tenant:
            body["tenant_id"] = self._tenant
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        url = f"{self._url}/v1/generate"
        log = active()
        log.debug(f"infrelay -> {kind} {ref}", kind=kind, route=str(ref),
                  tenant=self._tenant or None, url=url, timeout=timeout)
        log.trace(f"infrelay {kind} input", **shape(payload, max_depth=1))
        started = time.monotonic()
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            log.warning(f"infrelay {kind} {ref} unreachable: {exc}")
            raise InfrelayError(f"infrelay {kind}/{ref.provider} unreachable: {exc}") from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400:
            log.warning(f"infrelay {kind} {ref} failed ({response.status_code}) in {elapsed_ms}ms",
                        status=response.status_code, body=response.text[:300])
            raise InfrelayError(
                f"infrelay {kind}/{ref.provider} failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        result = response.json()
        log.debug(f"infrelay <- {kind} {ref} {response.status_code} in {elapsed_ms}ms",
                  status=response.status_code, ms=elapsed_ms,
                  output_type=(result.get("output") or {}).get("type"),
                  request_id=result.get("request_id"))
        return result

    def text(
        self, ref: ModelRef, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> Tuple[str, Dict]:
        """One chat completion: (text, usage). Usage carries model, token counts and
        finish_reason so a stage runner can detect a cut-off answer."""
        payload = {"system": system, "user": user,
                   "temperature": temperature, "max_tokens": max_tokens}
        output = self._generate("text", ref, payload, timeout=300).get("output") or {}
        if output.get("type") != "text":
            raise InfrelayError("infrelay text returned unsupported output")
        return str(output.get("value") or ""), dict(output.get("meta") or {})

    def complete(self, ref: ModelRef, prompt: str, *, max_tokens: int, temperature: float) -> str:
        return self.text(ref, "", prompt, temperature=temperature, max_tokens=max_tokens)[0]

    def image(self, ref: ModelRef, prompt: str, spec: ImageSpec) -> bytes:
        payload: Dict[str, Any] = {
            "prompt": prompt, "aspect_ratio": spec.aspect_ratio, "safe": not spec.mature,
        }
        if ref.model:
            payload["model"] = ref.model
        if spec.seed is not None:
            payload["seed"] = int(spec.seed)
        if spec.negative:
            payload["negative"] = spec.negative
        if spec.reference is not None:
            payload["reference_b64"] = base64.b64encode(spec.reference).decode()
        self._add_enhance(payload, spec.enhance, spec.enhance_style)
        return self._bytes(self._generate("image", ref, payload, timeout=600))

    def video(self, ref: ModelRef, prompt: str, spec: VideoSpec) -> bytes:
        """Video (mp4) bytes. A native-audio model voices `dialogue` itself; `music` toggles a
        non-diegetic score. Providers without their own audio ignore both."""
        payload: Dict[str, Any] = {
            "prompt": prompt, "seconds": spec.seconds, "resolution": spec.resolution,
            "aspect_ratio": spec.aspect_ratio, "safe": not spec.mature,
        }
        if ref.model:
            payload["model"] = ref.model
        if spec.image is not None:
            payload["reference_b64"] = base64.b64encode(spec.image).decode()
        self._add_enhance(payload, spec.enhance, spec.enhance_style)
        if spec.dialogue:
            payload["dialogue"] = spec.dialogue
        if not spec.music:
            payload["music"] = False
        return self._bytes(self._generate("video", ref, payload, timeout=1800))

    def music(self, ref: ModelRef, prompt: str, *, seconds: int = 30) -> bytes:
        payload: Dict[str, Any] = {"prompt": prompt, "seconds": seconds}
        if ref.model:
            payload["model"] = ref.model
        return self._bytes(self._generate("music", ref, payload, timeout=1800))

    def speech(self, ref: ModelRef, text: str, *, voice: str = "", language: str = "") -> bytes:
        payload: Dict[str, Any] = {"text": text}
        if voice:
            payload["voice"] = voice
        if language:
            payload["language"] = language
        return self._bytes(self._generate("audio", ref, payload, timeout=600))

    def transcribe(self, audio: bytes, ref: ModelRef, **kwargs: Any) -> Tuple[List[Dict], Dict]:
        payload: Dict[str, Any] = {"audio_b64": base64.b64encode(audio).decode()}
        if kwargs.get("language"):
            payload["language"] = kwargs["language"]
        if kwargs.get("word_timestamps"):
            payload["word_timestamps"] = True
        params = kwargs.get("params") or {}
        payload.update({key: value for key, value in params.items() if value is not None})
        output = self._generate("transcribe", ref, payload, timeout=1800).get("output") or {}
        if output.get("type") != "transcript":
            raise InfrelayError("infrelay transcribe returned unsupported output")
        meta = output.get("meta") or {}
        return list(meta.get("segments") or []), meta

    @staticmethod
    def _add_enhance(payload: Dict[str, Any], enhance: bool, style: str) -> None:
        if enhance:
            payload["enhance"] = True
            if style:
                payload["enhance_style"] = style

    @staticmethod
    def _bytes(result: Dict[str, Any]) -> bytes:
        output = result.get("output") or {}
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
