# tests/nodes/test_n4_section_processor_router.py
import pytest

from src.nodes.n4_section_processor_router import N4SectionProcessorRouter
from src.state import AgentState, HumanReviewFeedback, SectionDetail, SectionStatus


@pytest.fixture()
def router_node():
    return N4SectionProcessorRouter()


class TestSectionProcessorRouter:
    def test_route_to_process_next_section_when_pending_exists(
        self, router_node, caplog
    ):
        outline = [
            SectionDetail(
                id="1",
                title="S1",
                level=1,
                description_objectives="D1",
                original_requirements_summary="R1",
                status=SectionStatus.CONTENT_APPROVED,
            ),
            SectionDetail(
                id="2",
                title="S2",
                level=1,
                description_objectives="D2",
                original_requirements_summary="R2",
                status=SectionStatus.PENDING,
            ),
            SectionDetail(
                id="3",
                title="S3",
                level=1,
                description_objectives="D3",
                original_requirements_summary="R3",
                status=SectionStatus.PENDING,
            ),
        ]
        state = AgentState(thesis_outline=outline, current_section_index_for_router=0)
        result = router_node.run(state)

        assert result["next_node_override"] == "N5_ContextRetrievalNode"
        assert result["current_section_id"] == "2"
        assert result["current_section_index"] == 1
        assert "N4: Section 'S2' (ID: 2) is pending." in caplog.text

    def test_route_to_process_next_section_for_modification_requested(
        self, router_node, caplog
    ):
        outline = [
            SectionDetail(
                id="1",
                title="S1",
                level=1,
                description_objectives="D1",
                original_requirements_summary="R1",
                status=SectionStatus.CONTENT_APPROVED,
            ),
            SectionDetail(
                id="2",
                title="S2",
                level=1,
                description_objectives="D2",
                original_requirements_summary="R2",
                status=SectionStatus.HUMAN_REVIEW_PENDING,
                human_review_feedback=HumanReviewFeedback(
                    modification_requested=True, feedback_text="Needs changes"
                ),
            ),
            SectionDetail(
                id="3",
                title="S3",
                level=1,
                description_objectives="D3",
                original_requirements_summary="R3",
                status=SectionStatus.PENDING,
            ),
        ]
        state = AgentState(thesis_outline=outline, current_section_index_for_router=1)
        result = router_node.run(state)

        assert result["next_node_override"] == "N5_ContextRetrievalNode"
        assert result["current_section_id"] == "2"
        assert result["current_section_index"] == 1
        assert "N4: Section 'S2' (ID: 2) requires modification." in caplog.text

    def test_route_continues_from_last_index_if_no_modification_requested(
        self, router_node, caplog
    ):
        outline = [
            SectionDetail(
                id="1",
                title="S1",
                level=1,
                description_objectives="D1",
                original_requirements_summary="R1",
                status=SectionStatus.CONTENT_APPROVED,
            ),
            SectionDetail(
                id="2",
                title="S2",
                level=1,
                description_objectives="D2",
                original_requirements_summary="R2",
                status=SectionStatus.CONTENT_APPROVED,
                human_review_feedback=HumanReviewFeedback(modification_requested=False),
            ),
            SectionDetail(
                id="3",
                title="S3",
                level=1,
                description_objectives="D3",
                original_requirements_summary="R3",
                status=SectionStatus.PENDING,
            ),
        ]
        state = AgentState(thesis_outline=outline, current_section_index_for_router=1)
        result = router_node.run(state)

        assert result["next_node_override"] == "N5_ContextRetrievalNode"
        assert result["current_section_id"] == "3"
        assert result["current_section_index"] == 2
        assert "N4: Section 'S3' (ID: 3) is pending." in caplog.text

    def test_route_to_compile_when_all_done_or_error(self, router_node, caplog):
        outline = [
            SectionDetail(
                id="1",
                title="S1",
                level=1,
                description_objectives="D1",
                original_requirements_summary="R1",
                status=SectionStatus.CONTENT_APPROVED,
            ),
            SectionDetail(
                id="2",
                title="S2",
                level=1,
                description_objectives="D2",
                original_requirements_summary="R2",
                status=SectionStatus.ERROR,
            ),
            SectionDetail(
                id="3",
                title="S3",
                level=1,
                description_objectives="D3",
                original_requirements_summary="R3",
                status=SectionStatus.SKIPPED_BY_USER,
            ),
        ]
        state = AgentState(thesis_outline=outline, current_section_index_for_router=2)
        result = router_node.run(state)

        assert result["next_node_override"] == "N9_BibliographyManagerNode"
        assert "N4: Toutes les sections semblent traitées ou en erreur." in caplog.text

    def test_route_to_error_when_outline_empty(self, router_node, caplog):
        state = AgentState(thesis_outline=[])
        result = router_node.run(state)
        assert result["next_node_override"] == "ERROR_HANDLER"
        assert "N4: Thesis outline is empty." in caplog.text
        assert result["error_message"] == "N4: Thesis outline is empty."

    def test_finds_earlier_pending_section(self, router_node, caplog):
        outline = [
            SectionDetail(
                id="1",
                title="S1",
                level=1,
                description_objectives="D1",
                original_requirements_summary="R1",
                status=SectionStatus.PENDING,
            ),
            SectionDetail(
                id="2",
                title="S2",
                level=1,
                description_objectives="D2",
                original_requirements_summary="R2",
                status=SectionStatus.CONTENT_APPROVED,
            ),
            SectionDetail(
                id="3",
                title="S3",
                level=1,
                description_objectives="D3",
                original_requirements_summary="R3",
                status=SectionStatus.CONTENT_APPROVED,
            ),
        ]
        state = AgentState(thesis_outline=outline, current_section_index_for_router=1)
        result = router_node.run(state)

        assert result["next_node_override"] == "N5_ContextRetrievalNode"
        assert result["current_section_id"] == "1"
        assert result["current_section_index"] == 0
        assert result.get("current_section_index_for_router") == 0
        assert "N4: Found earlier PENDING section 'S1' (ID: 1)." in caplog.text

    def test_all_sections_not_final_but_no_pending_leads_to_warning_and_compile(
        self, router_node, caplog
    ):
        outline = [
            SectionDetail(
                id="1",
                title="S1",
                level=1,
                description_objectives="D1",
                original_requirements_summary="R1",
                status=SectionStatus.CONTENT_APPROVED,
            ),
            SectionDetail(
                id="2",
                title="S2",
                level=1,
                description_objectives="D2",
                original_requirements_summary="R2",
                status=SectionStatus.DRAFT_GENERATED,
            ),
        ]
        state = AgentState(thesis_outline=outline, current_section_index_for_router=1)
        result = router_node.run(state)

        assert result["next_node_override"] == "N9_BibliographyManagerNode"
        assert "N4: Inconsistent section states. Check thesis_outline." in result.get(
            "error_message", ""
        )
        assert (
            "N4: No PENDING sections found, but not all sections are in a final state."
            in caplog.text
        )
