from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Tuple

import httpx

from kinoforge.contract import AITask, ImageSpec, ModelRef, VideoSpec
from kinoforge.inference.charges import InferenceCharges
from kinoforge.observ import active, shape


class InfrelayError(RuntimeError):
    pass


class InfrelayClient:
    """Client for the Infrelay POST /v1/inference/{task} contract (cloud Infrelay or
    Infrelay-lite). Every call names its task; the pick says which model (or `auto`)."""

    # One connection pool for every call, so a story's many calls reuse warm connections
    # instead of paying a TCP and TLS handshake each.
    _http = httpx.Client(limits=httpx.Limits(max_connections=64, max_keepalive_connections=16))
    # Seconds between reads of a queued call: the first, and the most it backs off to.
    POLL_FIRST = 1.0
    POLL_MOST = 10.0

    def __init__(self, url: str, token: str = "", tenant: str = "") -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._tenant = tenant
        # Prompt rewrites infrelay ran for this client's calls, for the caller to bill.
        self.rewrites: List[Dict[str, Any]] = []
        # The provider and model that answered the last call; an auto pick only learns them
        # from the answer.
        self.served: Dict[str, str] = {}

    def _body(self, ref: ModelRef, payload: Dict[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"inference": ref.inference(), "input": payload}
        if self._tenant:
            body["tenant_id"] = self._tenant
        return body

    def _infer(
        self, task: AITask, ref: ModelRef, payload: Dict[str, Any], *, timeout: float
    ) -> Dict[str, Any]:
        log = active()
        if not self._url:
            raise InfrelayError("INFRELAY_URL is required")
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        url = f"{self._url}/v1/inference/{task.value}"
        log.debug(f"infrelay -> {task} {ref}", task=task.value, route=str(ref),
                  tenant=self._tenant or None, url=url, timeout=timeout,
                  payload=shape(payload, max_depth=2))
        log.trace(f"infrelay {task} input", **shape(payload, max_depth=1))
        started = time.monotonic()
        try:
            response = self._http.post(
                url, headers=headers, json=self._body(ref, payload), timeout=timeout
            )
        except httpx.HTTPError as exc:
            log.warning(f"infrelay {task} {ref} unreachable: {exc}")
            raise InfrelayError(f"infrelay {task} {ref} unreachable: {exc}") from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400 and response.status_code != 409:
            log.warning(f"infrelay {task} {ref} failed ({response.status_code}) in {elapsed_ms}ms",
                        status=response.status_code, body=response.text[:300])
            raise InfrelayError(
                f"infrelay {task} {ref} failed ({response.status_code}): {response.text[:300]}"
            )
        if response.status_code == 409:
            # A model (or the machine it runs on) that only runs queued: queue it and wait.
            log.debug(f"infrelay {task} {ref} runs queued: polling", detail=response.text[:200])
            return self._queued(task, ref, payload, timeout=timeout)
        result = response.json()
        self._answered(task, ref, result)
        log.debug(f"infrelay <- {task} {ref} {response.status_code} in {elapsed_ms}ms",
                  status=response.status_code, ms=elapsed_ms, served=self.served,
                  output_type=(result.get("output") or {}).get("type"),
                  request_id=result.get("request_id"))
        return result

    def _answered(self, task: AITask, ref: ModelRef, result: Dict[str, Any]) -> None:
        """Notes who served a call, what it cost (for the caller to bill) and its rewrite."""
        usage = result.get("usage") or {}
        self.served = {
            "provider": str(result.get("provider") or ref.provider),
            "model": str(result.get("model") or ref.model),
        }
        InferenceCharges.add({
            "task": task.value,
            "kind": str(result.get("kind") or ""),
            **self.served,
            "credits": int(usage.get("credits") or 0),
            "byok": bool(usage.get("byok")),
            "units": int(usage.get("units") or 1),
        })
        rewrite = usage.get("enhance")
        if isinstance(rewrite, dict):
            self.rewrites.append(dict(rewrite))

    def _queued(
        self, task: AITask, ref: ModelRef, payload: Dict[str, Any], *, timeout: float
    ) -> Dict[str, Any]:
        """Queues the call and reads the job until it ends, no faster than infrelay allows
        (its Retry-After)."""
        body = self._body(ref, payload)
        body["queue"] = True
        job_id = str(self._jobs("POST", f"/v1/inference/{task.value}", body=body)["id"])
        deadline = time.monotonic() + timeout
        wait = self.POLL_FIRST
        while True:
            time.sleep(wait)
            job, retry_after = self._read(job_id)
            if retry_after is not None:
                wait = retry_after
                continue
            status = job.get("status")
            if status == "done":
                result = dict(job.get("result") or {})
                self._answered(task, ref, result)
                return result
            if status in ("error", "cancelled"):
                raise InfrelayError(f"infrelay {task} {ref} failed: {job.get('error') or status}")
            if time.monotonic() > deadline:
                self.cancel_job(job_id)
                raise InfrelayError(f"infrelay {task} {ref} did not finish within {timeout}s")
            wait = min(wait * 1.5, self.POLL_MOST)

    def _read(self, job_id: str) -> Tuple[Dict[str, Any], float | None]:
        """The job, or how long infrelay asks to wait before reading it again."""
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            response = self._http.request(
                "GET", f"{self._url}/v1/jobs/{job_id}", headers=headers, json=None, timeout=60
            )
        except httpx.HTTPError as exc:
            raise InfrelayError(f"infrelay jobs unreachable: {exc}") from exc
        if response.status_code == 429:
            return {}, float(response.headers.get("Retry-After") or self.POLL_MOST)
        if response.status_code >= 400:
            raise InfrelayError(f"infrelay job read failed ({response.status_code})")
        return response.json(), None

    def text(
        self, task: AITask, ref: ModelRef, system: str, user: str, *, temperature: float,
        max_tokens: int,
    ) -> Tuple[str, Dict]:
        """One chat completion: (text, usage). Usage carries model, token counts and
        finish_reason so a stage runner can detect a cut-off answer."""
        payload = {"system": system, "user": user,
                   "temperature": temperature, "max_tokens": max_tokens}
        output = self._infer(task, ref, payload, timeout=300).get("output") or {}
        if output.get("type") != "text":
            raise InfrelayError("infrelay text returned unsupported output")
        return str(output.get("value") or ""), dict(output.get("meta") or {})

    def complete(self, task: AITask, ref: ModelRef, prompt: str, *, max_tokens: int,
                 temperature: float, system: str = "") -> str:
        return self.text(
            task, ref, system, prompt, temperature=temperature, max_tokens=max_tokens
        )[0]

    def image(self, task: AITask, ref: ModelRef, prompt: str, spec: ImageSpec) -> bytes:
        payload: Dict[str, Any] = {
            "prompt": prompt, "aspect_ratio": spec.aspect_ratio, "safe": not spec.mature,
        }
        if spec.seed is not None:
            payload["seed"] = int(spec.seed)
        if spec.negative:
            payload["negative"] = spec.negative
        if spec.reference is not None:
            payload["reference_b64"] = base64.b64encode(spec.reference).decode()
        self._add_enhance(payload, spec.enhance, spec.enhance_style)
        return self._bytes(self._infer(task, ref, payload, timeout=1000))

    def video(self, task: AITask, ref: ModelRef, prompt: str, spec: VideoSpec) -> bytes:
        """Video (mp4) bytes. A native-audio model voices `dialogue` itself; `music` toggles a
        non-diegetic score. Providers without their own audio ignore both."""
        payload = self._video_payload(prompt, spec)
        return self._bytes(self._infer(task, ref, payload, timeout=1800))

    def submit_video(
        self, task: AITask, ref: ModelRef, prompt: str, spec: VideoSpec, callback_url: str
    ) -> str:
        """Queues a clip on infrelay and returns its job id at once. Infrelay posts the job to
        `callback_url` when it ends; read it with job and job_media."""
        body = self._body(ref, self._video_payload(prompt, spec))
        body["callback_url"] = callback_url
        job = self._jobs("POST", f"/v1/inference/{task.value}", body=body)
        active().debug(f"infrelay queued video {ref}", job=job.get("id"), status=job.get("status"))
        return str(job["id"])

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._jobs("GET", f"/v1/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self._jobs("DELETE", f"/v1/jobs/{job_id}")

    def job_media(self, job: Dict[str, Any]) -> bytes:
        """A finished job's media bytes; what it cost and its prompt rewrite are kept for the
        caller to bill."""
        result = job.get("result") or {}
        task = AITask(str(job.get("task") or result.get("task") or job.get("kind") or "video"))
        self._answered(task, ModelRef(str(job.get("provider") or ""), str(job.get("model") or "")),
                       result)
        return self._bytes(result)

    def _jobs(
        self, method: str, path: str, *, body: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        if not self._url:
            raise InfrelayError("INFRELAY_URL is required")
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            response = self._http.request(method, f"{self._url}{path}", headers=headers,
                                          json=body, timeout=60)
        except httpx.HTTPError as exc:
            raise InfrelayError(f"infrelay jobs unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise InfrelayError(
                f"infrelay job {method} failed ({response.status_code}): {response.text[:300]}"
            )
        return response.json()

    def _video_payload(self, prompt: str, spec: VideoSpec) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt, "seconds": spec.seconds, "resolution": spec.resolution,
            "aspect_ratio": spec.aspect_ratio, "safe": not spec.mature,
        }
        if spec.image is not None:
            payload["reference_b64"] = base64.b64encode(spec.image).decode()
        self._add_enhance(payload, spec.enhance, spec.enhance_style)
        if spec.dialogue:
            payload["dialogue"] = spec.dialogue
        if not spec.music:
            payload["music"] = False
        return payload

    def music(self, task: AITask, ref: ModelRef, prompt: str, *, seconds: int = 30) -> bytes:
        payload: Dict[str, Any] = {"prompt": prompt, "seconds": seconds}
        return self._bytes(self._infer(task, ref, payload, timeout=1800))

    def speech(self, task: AITask, ref: ModelRef, text: str, *, voice: str = "",
               language: str = "") -> bytes:
        payload: Dict[str, Any] = {"text": text}
        if voice:
            payload["voice"] = voice
        if language:
            payload["language"] = language
        return self._bytes(self._infer(task, ref, payload, timeout=600))

    def transcribe(self, task: AITask, audio: bytes, ref: ModelRef,
                   **kwargs: Any) -> Tuple[List[Dict], Dict]:
        payload: Dict[str, Any] = {"audio_b64": base64.b64encode(audio).decode()}
        if kwargs.get("language"):
            payload["language"] = kwargs["language"]
        if kwargs.get("word_timestamps"):
            payload["word_timestamps"] = True
        params = kwargs.get("params") or {}
        payload.update({key: value for key, value in params.items() if value is not None})
        output = self._infer(task, ref, payload, timeout=1800).get("output") or {}
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
                response = InfrelayClient._http.get(
                    str(output["value"]), timeout=600, follow_redirects=True
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise InfrelayError(f"could not fetch generated media: {exc}") from exc
            return response.content
        raise InfrelayError(f"infrelay returned unsupported output type: {kind}")
