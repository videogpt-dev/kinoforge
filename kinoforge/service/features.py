from kinoforge.contract import JobKind
from kinoforge.service.models import SegmentFeatureResponse, SegmentStatus


class FeatureCatalog:
    @staticmethod
    def for_segment(code_name: JobKind) -> list[SegmentFeatureResponse]:
        status = SegmentStatus.AVAILABLE
        definitions = {
            JobKind.CLIPS: (
                (
                    "transcription",
                    "Transcription",
                    "Create multilingual timed transcripts through Infrelay.",
                    "audio-lines",
                ),
                (
                    "viral-moments",
                    "Viral moment discovery",
                    "Find strong short-form moments using AI or offline analysis.",
                    "scan-search",
                ),
                (
                    "moment-ranking",
                    "Moment ranking",
                    "Score and rank moments using hook and engagement signals.",
                    "chart-no-axes-column-increasing",
                ),
                (
                    "clip-rendering",
                    "Clip rendering",
                    "Cut selected moments into standalone video artifacts.",
                    "film",
                ),
                (
                    "multi-format",
                    "Multi-format output",
                    "Create portrait, landscape, and platform-ready aspect variants.",
                    "panels-top-left",
                ),
                (
                    "captions",
                    "Caption alignment",
                    "Align canonical text to spoken word timing for accurate captions.",
                    "captions",
                ),
            ),
            JobKind.STORY: (
                (
                    "story-writing",
                    "Story writing",
                    "Turn an idea or script into a structured narrative.",
                    "pen-line",
                ),
                (
                    "scene-planning",
                    "Scene planning",
                    "Break stories into directed scenes with narration and visual prompts.",
                    "layout-list",
                ),
                (
                    "character-consistency",
                    "Character consistency",
                    "Maintain reusable character identity across generated scenes.",
                    "users",
                ),
                (
                    "media-generation",
                    "Scene media generation",
                    "Generate images, video, voiceover, and music through Infrelay.",
                    "wand-sparkles",
                ),
                (
                    "video-assembly",
                    "Video assembly",
                    "Assemble scenes, narration, music, captions, and transitions.",
                    "clapperboard",
                ),
            ),
            JobKind.SERIES: (
                (
                    "series-planning",
                    "Series planning",
                    "Define a reusable premise, visual identity, and episode structure.",
                    "notebook-tabs",
                ),
                (
                    "recurring-cast",
                    "Recurring cast",
                    "Reuse character identity and personality across episodes.",
                    "contact-round",
                ),
                (
                    "episode-planning",
                    "Episode planning",
                    "Generate connected episode ideas from the series premise.",
                    "list-video",
                ),
                (
                    "continuity",
                    "Series continuity",
                    "Carry story, character, and visual context between episodes.",
                    "git-branch",
                ),
                (
                    "episode-generation",
                    "Episode generation",
                    "Generate each episode through the story and media pipeline.",
                    "play-square",
                ),
            ),
        }
        return [
            SegmentFeatureResponse(
                code_name=feature[0],
                name=feature[1],
                description=feature[2],
                icon=feature[3],
                status=(
                    SegmentStatus.PLANNED
                    if code_name is JobKind.STORY and feature[0] == "video-assembly"
                    else status
                ),
            )
            for feature in definitions[code_name]
        ]
