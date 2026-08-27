from pathlib import Path
from typing import Any, Dict, List, Optional


class ExecutionStore:
    def __init__(
        self,
        root: Path,
        job_id: str,
        source: Path,
        *,
        record: Optional[Dict[str, Any]] = None,
        transcript: Optional[List[Dict[str, Any]]] = None,
        has_clips: bool = False,
    ) -> None:
        self._root = root
        self._job_id = job_id
        self._source = source
        self._record = record
        self._transcript = transcript
        self._has_clips = has_clips
        self._clips: List[Dict[str, Any]] = []
        self._root.mkdir(parents=True, exist_ok=True)

    def _check(self, job_id: str) -> None:
        if job_id != self._job_id:
            raise KeyError(f"execution store is bound to {self._job_id}, not {job_id}")

    @property
    def record(self) -> Optional[Dict[str, Any]]:
        return self._record

    @property
    def transcript(self) -> Optional[List[Dict[str, Any]]]:
        return self._transcript

    def workdir(self, job_id: str) -> Path:
        self._check(job_id)
        return self._root

    def save_record(self, job_id: str, record: Dict[str, Any]) -> None:
        self._check(job_id)
        self._record = record

    def load_record(self, job_id: str) -> Optional[Dict[str, Any]]:
        self._check(job_id)
        return self._record

    def source_path(self, job_id: str) -> Optional[Path]:
        self._check(job_id)
        return self._source

    def transcript_path(self, job_id: str) -> Path:
        self._check(job_id)
        return self._root / "transcript.json"

    def save_transcript(self, job_id: str, transcript: List[Dict[str, Any]]) -> None:
        self._check(job_id)
        self._transcript = transcript

    def load_transcript(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        self._check(job_id)
        return self._transcript

    def clips_path(self, job_id: str) -> Path:
        self._check(job_id)
        return self._root / "artifacts"

    def list_clips(self, job_id: str) -> List[Dict[str, Any]]:
        self._check(job_id)
        if self._has_clips and not self._clips:
            return [{"clip_id": "existing"}]
        return list(self._clips)

    def append_clips(
        self,
        job_id: str,
        project: Dict[str, Any],
        moments: List[Dict[str, Any]],
    ) -> List[Optional[str]]:
        self._check(job_id)
        created: List[Optional[str]] = []
        for index, moment in enumerate(moments, 1):
            if float(moment.get("end") or 0) <= float(moment.get("start") or 0):
                created.append(None)
                continue
            clip_id = f"clip_{index:02d}"
            self._clips.append({"clip_id": clip_id})
            created.append(clip_id)
        return created

    def clip_path(self, job_id: str, clip_id: str) -> Path:
        self._check(job_id)
        return self.clips_path(job_id) / clip_id

    def mark_rendered(
        self,
        job_id: str,
        clip_id: str,
        file: str = "clip.mp4",
        transcript: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._check(job_id)
