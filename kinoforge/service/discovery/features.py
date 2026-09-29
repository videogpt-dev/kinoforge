from kinoforge.contract import JobKind
from kinoforge.schemas import SegmentFeatureResponse
from kinoforge.service.discovery.catalog import spec_for


class FeatureCatalog:
    @staticmethod
    def for_segment(code_name: JobKind) -> list[SegmentFeatureResponse]:
        return [
            SegmentFeatureResponse(
                code_name=f.code_name, name=f.name, description=f.description,
                icon=f.icon, status=f.status,
            )
            for f in spec_for(code_name).features
        ]
