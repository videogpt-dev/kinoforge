from kinoforge.schemas import (
    SegmentDefinitionRequirementResponse,
    SegmentResponse,
    SegmentsResponse,
)
from kinoforge.service.discovery.catalog import CATALOG, SegmentSpec
from kinoforge.service.discovery.features import FeatureCatalog


class SegmentCatalog:
    @staticmethod
    def list() -> SegmentsResponse:
        return SegmentsResponse(segments=[SegmentCatalog._response(spec) for spec in CATALOG])

    @staticmethod
    def _response(spec: SegmentSpec) -> SegmentResponse:
        return SegmentResponse(
            id=spec.kind,
            code_name=spec.kind,
            name=spec.name,
            label=spec.label,
            description=spec.description,
            icon=spec.icon,
            status=spec.status,
            features=FeatureCatalog.for_segment(spec.kind),
            definition_requirements=[
                SegmentDefinitionRequirementResponse(type=r.type, key=r.key, purpose=r.purpose)
                for r in spec.requirements
            ],
            api_version="v1",
            execute_path=spec.execute_path,
        )
