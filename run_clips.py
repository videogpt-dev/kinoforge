"""Run the clips pipeline from a terminal, without core.

    python run_clips.py VIDEO [--transcript t.json] [--model provider/model] [--render]
    python run_clips.py --request saved_request.json

Transcription and the AI finder call Infrelay at $INFRELAY_URL, so run it inside the
kinoforge container (docker compose exec kinoforge python run_clips.py ...) or pass
--transcript and leave --model off to stay fully offline.
"""

import argparse
import json
import uuid
from pathlib import Path

from kinoforge.schemas import ClipsExecutionRequest
from kinoforge.service.runtimes import ClipsRuntime

parser = argparse.ArgumentParser()
parser.add_argument("video", nargs="?")
parser.add_argument("--request", help="replay a full request JSON (e.g. one core sent)")
parser.add_argument("--transcript", help="segments JSON [{start, end, text}]; skips Whisper")
parser.add_argument("--model", help="provider/model for the AI finder; offline when omitted")
parser.add_argument("--definitions", help="definition bundle JSON holding the AI prompts")
parser.add_argument("--render", action="store_true", help="cut clips too, not just find")
parser.add_argument("--count", type=int, default=5)
parser.add_argument("--min", type=float, default=20)
parser.add_argument("--max", type=float, default=60)
parser.add_argument("--out", default="/tmp/kinoforge-run")
parser.add_argument("--log", default="debug")
args = parser.parse_args()

if args.request:
    request = json.loads(Path(args.request).read_text())
else:
    if not args.video:
        parser.error("VIDEO or --request is required")
    provider, _, model = (args.model or "").partition("/")
    request = {
        "job_id": uuid.uuid4().hex,
        "project_id": Path(args.video).stem,
        "workspace": str(Path(args.out) / Path(args.video).stem),
        "input": {"video_path": str(Path(args.video).resolve())},
        "options": {
            "clip_count": args.count, "min_length": args.min, "max_length": args.max,
            "analyze_only": not args.render, "min_interest_score": 0,
        },
        "config": {
            "transcription": {"model": "base", "device": "cpu", "compute_type": "int8"},
            "moment_finder": "ai" if args.model else "offline",
            "moment_route": {"provider": provider, "model": model} if args.model else {},
            "slug": Path(args.video).stem,
            "output_dir": args.out,
            "processing": {"max_workers": 2, "use_gpu": False},
            "rendering": {"burn_subtitles": True, "mute_output": False},
        },
        "definitions": json.loads(Path(args.definitions).read_text()) if args.definitions else {
            "schema_version": 1, "id": "cli", "version": "sha256:cli",
            "engine": {"minimum": "0.1.0"}, "definitions": [],
        },
        "state": {"transcript": json.loads(Path(args.transcript).read_text())}
        if args.transcript else {},
    }
request["log_level"] = args.log

response = ClipsRuntime.from_env().execute(ClipsExecutionRequest.model_validate(request))

result = response["result"]
print(f"\nstatus: {result['status']}  {result.get('error') or ''}")
for moment in result["data"].get("used_moments") or []:
    print(f"  {moment['start']:7.1f}-{moment['end']:7.1f}  score={moment.get('score', 0):5.1f}  "
          f"{str(moment.get('text', ''))[:80]}")
for artifact in result.get("artifacts") or []:
    print(f"  clip: {artifact['path']}")
dump = Path(request["workspace"]) / "response.json"
dump.parent.mkdir(parents=True, exist_ok=True)
dump.write_text(json.dumps(response, indent=2, default=str))
print(f"full response: {dump}")
