from kinoforge.contract import JobKind
from kinoforge.service.features import FeatureCatalog
from kinoforge.service.models import (
    SegmentDefinitionRequirementResponse,
    SegmentResponse,
    SegmentsResponse,
    SegmentStatus,
)


class SegmentCatalog:
    @staticmethod
    def list() -> SegmentsResponse:
        return SegmentsResponse(
            segments=[
                SegmentResponse(
                    id=JobKind.CLIPS,
                    code_name=JobKind.CLIPS,
                    name="Clips",
                    label="Viral clip generator",
                    description=(
                        "Find, rank, render, and format viral short-form moments from long video."
                    ),
                    icon="scissors",
                    status=SegmentStatus.AVAILABLE,
                    features=FeatureCatalog.for_segment(JobKind.CLIPS),
                    definition_requirements=[
                        SegmentDefinitionRequirementResponse(
                            type="agent",
                            key="prompts.agents.moment_discovery",
                            purpose="Discover strongest moments from full transcript.",
                        ),
                        SegmentDefinitionRequirementResponse(
                            type="agent",
                            key="prompts.agents.clip_analysis",
                            purpose="Score candidates and refine clip boundaries.",
                        ),
                    ],
                    api_version="v1",
                    execute_path="/v1/segments/clips/execute",
                ),
                SegmentResponse(
                    id=JobKind.STORY,
                    code_name=JobKind.STORY,
                    name="Story",
                    label="Story video generator",
                    description="Generate a complete narrative video from a prompt or script.",
                    icon="clapperboard",
                    status=SegmentStatus.AVAILABLE,
                    features=FeatureCatalog.for_segment(JobKind.STORY),
                    definition_requirements=[
                        SegmentDefinitionRequirementResponse(
                            type="agent",
                            key="story_system",
                            purpose="Provide fallback screenwriter persona.",
                        ),
                        SegmentDefinitionRequirementResponse(
                            type="agent",
                            key="story_idea",
                            purpose="Turn project brief into screenwriter input.",
                        ),
                        SegmentDefinitionRequirementResponse(
                            type="fragment",
                            key="prompts.fragments.contract",
                            purpose="Define structured story output contract.",
                        ),
                        SegmentDefinitionRequirementResponse(
                            type="fragment",
                            key="prompts.fragments.language",
                            purpose="Apply requested narration language.",
                        ),
                    ],
                    api_version="v1",
                    execute_path="/v1/segments/story/write",
                ),
                SegmentResponse(
                    id=JobKind.SERIES,
                    code_name=JobKind.SERIES,
                    name="Series",
                    label="Series generator",
                    description="Create recurring characters and connected multi-episode videos.",
                    icon="list-video",
                    status=SegmentStatus.AVAILABLE,
                    features=FeatureCatalog.for_segment(JobKind.SERIES),
                    definition_requirements=[
                        SegmentDefinitionRequirementResponse(
                            type="agent",
                            key="episode_plan",
                            purpose="Propose connected episode ideas from the series premise.",
                        ),
                        SegmentDefinitionRequirementResponse(
                            type="agent",
                            key="story_system",
                            purpose="Provide fallback screenwriter persona for planning.",
                        ),
                    ],
                    api_version="v1",
                    execute_path="/v1/segments/series/plan",
                ),
            ]
        )
