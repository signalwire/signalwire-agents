"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for ConciergeAgent prefab
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, call


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# Minimal configuration shared across tests
VENUE_NAME = "Grand Hotel"
SERVICES = ["room service", "spa bookings", "restaurant reservations"]
AMENITIES = {
    "pool": {"hours": "7 AM - 10 PM", "location": "2nd Floor"},
    "gym": {"hours": "24 hours", "location": "3rd Floor"},
}


def _make_concierge(**overrides):
    """
    Create a ConciergeAgent with AgentBase.__init__ mocked out.

    By patching AgentBase.__init__ as a no-op we avoid the full constructor
    chain (schema files, uvicorn, POM imports, etc.) while still exercising
    all of ConciergeAgent's own initialisation logic.
    """
    with patch("signalwire_agents.prefabs.concierge.AgentBase.__init__", return_value=None) as mock_init:
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        # Stub the methods that _setup_concierge_agent calls on `self`
        # so that assertions can be made against them.
        with patch.multiple(
            ConciergeAgent,
            prompt_add_section=Mock(),
            set_post_prompt=Mock(),
            add_hints=Mock(),
            set_params=Mock(),
            set_global_data=Mock(),
            set_native_functions=Mock(),
        ):
            kwargs = dict(
                venue_name=VENUE_NAME,
                services=SERVICES,
                amenities=AMENITIES,
            )
            kwargs.update(overrides)
            agent = ConciergeAgent(**kwargs)

    return agent, mock_init


def _make_bare_concierge():
    """
    Create a ConciergeAgent using __new__ (skipping __init__) and set
    only the attributes needed by the tool methods under test.
    """
    from signalwire_agents.prefabs.concierge import ConciergeAgent

    agent = ConciergeAgent.__new__(ConciergeAgent)
    agent.venue_name = VENUE_NAME
    agent.services = list(SERVICES)
    agent.amenities = dict(AMENITIES)
    agent.hours_of_operation = {"default": "9 AM - 5 PM"}
    agent.special_instructions = []
    return agent


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestConciergeInitialization:
    """Test ConciergeAgent.__init__ delegates properly to AgentBase."""

    def test_super_init_called_with_defaults(self):
        """super().__init__ receives name, route, and use_pom."""
        agent, mock_init = _make_concierge()

        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        assert kwargs["name"] == "concierge"
        assert kwargs["route"] == "/concierge"
        assert kwargs["use_pom"] is True

    def test_super_init_custom_name_and_route(self):
        """Custom name and route are forwarded to AgentBase."""
        agent, mock_init = _make_concierge(name="lobby", route="/lobby")

        _, kwargs = mock_init.call_args
        assert kwargs["name"] == "lobby"
        assert kwargs["route"] == "/lobby"

    def test_extra_kwargs_forwarded_to_super(self):
        """Arbitrary **kwargs are forwarded to AgentBase.__init__."""
        agent, mock_init = _make_concierge(host="0.0.0.0", port=9090)

        _, kwargs = mock_init.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9090

    def test_venue_name_stored(self):
        agent, _ = _make_concierge()
        assert agent.venue_name == VENUE_NAME

    def test_services_stored(self):
        agent, _ = _make_concierge()
        assert agent.services == SERVICES

    def test_amenities_stored(self):
        agent, _ = _make_concierge()
        assert agent.amenities == AMENITIES

    def test_default_hours_of_operation(self):
        """When no hours supplied, a sensible default is used."""
        agent, _ = _make_concierge()
        assert agent.hours_of_operation == {"default": "9 AM - 5 PM"}

    def test_custom_hours_of_operation(self):
        custom_hours = {"weekdays": "8 AM - 8 PM", "weekends": "10 AM - 6 PM"}
        agent, _ = _make_concierge(hours_of_operation=custom_hours)
        assert agent.hours_of_operation == custom_hours

    def test_default_special_instructions_empty(self):
        agent, _ = _make_concierge()
        assert agent.special_instructions == []

    def test_custom_special_instructions(self):
        instructions = ["Always upsell the premium package", "Mention the loyalty program"]
        agent, _ = _make_concierge(special_instructions=instructions)
        assert agent.special_instructions == instructions


# ---------------------------------------------------------------------------
# _setup_concierge_agent
# ---------------------------------------------------------------------------


class TestSetupConciergeAgent:
    """Verify that _setup_concierge_agent configures the agent correctly."""

    @pytest.fixture(autouse=True)
    def _create_agent(self):
        """Create a bare agent and mock out the AgentBase helper methods."""
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        self.agent = ConciergeAgent.__new__(ConciergeAgent)
        self.agent.venue_name = VENUE_NAME
        self.agent.services = list(SERVICES)
        self.agent.amenities = dict(AMENITIES)
        self.agent.hours_of_operation = {"default": "9 AM - 5 PM"}
        self.agent.special_instructions = []

        # Mock every method called by _setup_concierge_agent
        self.agent.prompt_add_section = Mock()
        self.agent.set_post_prompt = Mock()
        self.agent.add_hints = Mock()
        self.agent.set_params = Mock()
        self.agent.set_global_data = Mock()
        self.agent.set_native_functions = Mock()

    # -- Personality section --------------------------------------------------

    def test_personality_section_added(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        personality_calls = [c for c in calls if c[0][0] == "Personality"]
        assert len(personality_calls) == 1
        assert VENUE_NAME in personality_calls[0][1]["body"]

    def test_goal_section_added(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        goal_calls = [c for c in calls if c[0][0] == "Goal"]
        assert len(goal_calls) == 1
        assert "exceptional service" in goal_calls[0][1]["body"].lower()

    # -- Instructions section -------------------------------------------------

    def test_instructions_section_has_base_bullets(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        instr_calls = [c for c in calls if c[0][0] == "Instructions"]
        assert len(instr_calls) == 1
        bullets = instr_calls[0][1]["bullets"]
        assert len(bullets) >= 4  # at least the 4 base instructions

    def test_special_instructions_appended(self):
        self.agent.special_instructions = ["Speak only French"]
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        instr_calls = [c for c in calls if c[0][0] == "Instructions"]
        bullets = instr_calls[0][1]["bullets"]
        assert "Speak only French" in bullets

    # -- Services section -----------------------------------------------------

    def test_services_section_lists_all_services(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        svc_calls = [c for c in calls if c[0][0] == "Available Services"]
        assert len(svc_calls) == 1
        body = svc_calls[0][1]["body"]
        for svc in SERVICES:
            assert svc in body

    # -- Amenities section ----------------------------------------------------

    def test_amenities_section_has_subsections(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        amen_calls = [c for c in calls if c[0][0] == "Amenities"]
        assert len(amen_calls) == 1
        subsections = amen_calls[0][1]["subsections"]
        titles = {s["title"] for s in subsections}
        assert "Pool" in titles
        assert "Gym" in titles

    def test_amenities_subsection_body_contains_details(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        amen_calls = [c for c in calls if c[0][0] == "Amenities"]
        subsections = amen_calls[0][1]["subsections"]
        pool_sub = [s for s in subsections if s["title"] == "Pool"][0]
        assert "7 AM - 10 PM" in pool_sub["body"]
        assert "2nd Floor" in pool_sub["body"]

    # -- Hours section --------------------------------------------------------

    def test_hours_of_operation_section(self):
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        hour_calls = [c for c in calls if c[0][0] == "Hours of Operation"]
        assert len(hour_calls) == 1
        assert "9 AM - 5 PM" in hour_calls[0][1]["body"]

    def test_hours_section_with_custom_hours(self):
        self.agent.hours_of_operation = {
            "weekdays": "8 AM - 9 PM",
            "weekends": "10 AM - 6 PM",
        }
        self.agent._setup_concierge_agent()

        calls = self.agent.prompt_add_section.call_args_list
        hour_calls = [c for c in calls if c[0][0] == "Hours of Operation"]
        body = hour_calls[0][1]["body"]
        assert "8 AM - 9 PM" in body
        assert "10 AM - 6 PM" in body

    # -- Post-prompt ----------------------------------------------------------

    def test_post_prompt_set(self):
        self.agent._setup_concierge_agent()
        self.agent.set_post_prompt.assert_called_once()
        prompt_text = self.agent.set_post_prompt.call_args[0][0]
        assert "topic" in prompt_text
        assert "follow_up_needed" in prompt_text

    # -- Hints ----------------------------------------------------------------

    def test_hints_include_venue_and_services_and_amenities(self):
        self.agent._setup_concierge_agent()
        self.agent.add_hints.assert_called_once()
        hints = self.agent.add_hints.call_args[0][0]
        assert VENUE_NAME in hints
        for svc in SERVICES:
            assert svc in hints
        for key in AMENITIES:
            assert key in hints

    # -- Params ---------------------------------------------------------------

    def test_default_params_set(self):
        self.agent._setup_concierge_agent()

        # set_params may be called once (no welcome) or twice (with welcome).
        first_call_params = self.agent.set_params.call_args_list[0][0][0]
        assert first_call_params["wait_for_user"] is False
        assert first_call_params["end_of_speech_timeout"] == 1000
        assert first_call_params["ai_volume"] == 5
        assert first_call_params["local_tz"] == "America/New_York"

    # -- Global data ----------------------------------------------------------

    def test_global_data_set(self):
        self.agent._setup_concierge_agent()
        self.agent.set_global_data.assert_called_once()
        data = self.agent.set_global_data.call_args[0][0]
        assert data["venue_name"] == VENUE_NAME
        assert data["services"] == SERVICES
        assert data["amenities"] == AMENITIES
        assert data["hours"] == {"default": "9 AM - 5 PM"}

    # -- Native functions -----------------------------------------------------

    def test_native_functions_set(self):
        self.agent._setup_concierge_agent()
        self.agent.set_native_functions.assert_called_once_with(["check_time"])

    # -- Welcome message ------------------------------------------------------

    def test_no_welcome_message_no_static_greeting(self):
        self.agent._setup_concierge_agent(welcome_message=None)
        # set_params called exactly once (the default call)
        assert self.agent.set_params.call_count == 1

    def test_welcome_message_sets_static_greeting(self):
        self.agent._setup_concierge_agent(welcome_message="Welcome to the Grand Hotel!")
        # set_params should be called twice: default + greeting override
        assert self.agent.set_params.call_count == 2
        greeting_call = self.agent.set_params.call_args_list[1][0][0]
        assert greeting_call["static_greeting"] == "Welcome to the Grand Hotel!"
        assert greeting_call["static_greeting_no_barge"] is True


# ---------------------------------------------------------------------------
# check_availability tool
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    """Test the check_availability SWAIG tool method."""

    def setup_method(self):
        self.agent = _make_bare_concierge()

    def test_known_service_returns_available(self):
        from signalwire_agents.core.function_result import SwaigFunctionResult

        result = self.agent.check_availability(
            {"service": "spa bookings", "date": "2025-03-15", "time": "14:00"},
            raw_data={},
        )
        assert isinstance(result, SwaigFunctionResult)
        assert "spa bookings" in result.response
        assert "2025-03-15" in result.response
        assert "14:00" in result.response

    def test_known_service_case_insensitive(self):
        from signalwire_agents.core.function_result import SwaigFunctionResult

        result = self.agent.check_availability(
            {"service": "Room Service", "date": "2025-01-01", "time": "08:00"},
            raw_data={},
        )
        assert isinstance(result, SwaigFunctionResult)
        # The lowered input "room service" matches "room service" in SERVICES
        assert "room service" in result.response.lower()
        assert "available" in result.response.lower() or "reservation" in result.response.lower()

    def test_unknown_service_returns_error(self):
        from signalwire_agents.core.function_result import SwaigFunctionResult

        result = self.agent.check_availability(
            {"service": "helicopter tours", "date": "2025-06-01", "time": "10:00"},
            raw_data={},
        )
        assert isinstance(result, SwaigFunctionResult)
        assert "sorry" in result.response.lower() or "don't offer" in result.response.lower()
        assert VENUE_NAME in result.response
        # Should list available services
        for svc in SERVICES:
            assert svc in result.response

    def test_missing_service_arg_treated_as_unknown(self):
        """Empty service defaults to empty string, which won't match."""
        result = self.agent.check_availability(
            {"date": "2025-01-01", "time": "09:00"},
            raw_data={},
        )
        # Empty string won't be in services list, so we get the "don't offer" path
        assert VENUE_NAME in result.response

    def test_missing_date_and_time_still_works(self):
        """Even without date/time, availability for a known service works."""
        result = self.agent.check_availability(
            {"service": "room service"},
            raw_data={},
        )
        assert "room service" in result.response.lower()

    def test_raw_data_passed_through(self):
        """raw_data is accepted but not used internally."""
        sentinel = {"call_id": "abc-123"}
        # Should not raise
        result = self.agent.check_availability(
            {"service": "spa bookings", "date": "2025-01-01", "time": "12:00"},
            raw_data=sentinel,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# get_directions tool
# ---------------------------------------------------------------------------


class TestGetDirections:
    """Test the get_directions SWAIG tool method."""

    def setup_method(self):
        self.agent = _make_bare_concierge()

    def test_known_amenity_with_location(self):
        from signalwire_agents.core.function_result import SwaigFunctionResult

        result = self.agent.get_directions({"location": "pool"}, raw_data={})
        assert isinstance(result, SwaigFunctionResult)
        assert "2nd Floor" in result.response

    def test_known_amenity_gym(self):
        result = self.agent.get_directions({"location": "gym"}, raw_data={})
        assert "3rd Floor" in result.response

    def test_unknown_location_returns_fallback(self):
        result = self.agent.get_directions({"location": "helipad"}, raw_data={})
        assert "front desk" in result.response.lower() or "don't have" in result.response.lower()

    def test_amenity_without_location_field(self):
        """If an amenity exists but has no 'location' key, fallback is used."""
        self.agent.amenities["sauna"] = {"hours": "10 AM - 8 PM"}  # no "location"
        result = self.agent.get_directions({"location": "sauna"}, raw_data={})
        # Should hit the else branch because "location" not in details
        assert "don't have" in result.response.lower() or "front desk" in result.response.lower()

    def test_empty_location_arg(self):
        result = self.agent.get_directions({}, raw_data={})
        # Empty string won't match any amenity
        assert result is not None

    def test_location_case_sensitivity(self):
        """Location lookup is lowercased; amenity keys in our fixture are lowercase."""
        result = self.agent.get_directions({"location": "Pool"}, raw_data={})
        # "Pool".lower() == "pool" which exists in amenities
        assert "2nd Floor" in result.response

    def test_raw_data_accepted(self):
        sentinel = {"meta": "data"}
        result = self.agent.get_directions({"location": "pool"}, raw_data=sentinel)
        assert result is not None


# ---------------------------------------------------------------------------
# on_summary
# ---------------------------------------------------------------------------


class TestOnSummary:
    """Test the on_summary callback."""

    def setup_method(self):
        self.agent = _make_bare_concierge()

    def test_dict_summary_printed(self, capsys):
        summary = {"topic": "pool hours", "follow_up_needed": False}
        self.agent.on_summary(summary)
        captured = capsys.readouterr()
        assert "Concierge interaction summary" in captured.out
        assert "pool hours" in captured.out

    def test_string_summary_printed(self, capsys):
        self.agent.on_summary("Guest asked about the gym")
        captured = capsys.readouterr()
        assert "Guest asked about the gym" in captured.out

    def test_none_summary_no_output(self, capsys):
        self.agent.on_summary(None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_empty_string_summary_no_output(self, capsys):
        """Empty string is falsy, so no output should be produced."""
        self.agent.on_summary("")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_empty_dict_summary_treated_as_falsy(self, capsys):
        """An empty dict is falsy; no output expected."""
        self.agent.on_summary({})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_raw_data_accepted(self, capsys):
        """on_summary accepts an optional raw_data kwarg."""
        self.agent.on_summary("test", raw_data={"call_id": "xyz"})
        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_exception_during_summary_processing(self, capsys):
        """If json.dumps raises, the except branch prints the error."""
        bad_summary = Mock()
        bad_summary.__bool__ = Mock(return_value=True)
        # isinstance check for dict will return False for Mock, so it goes
        # to the else branch which calls print(f"... {summary}").  That
        # won't raise, so let's force isinstance to return True and make
        # json.dumps fail.
        with patch("signalwire_agents.prefabs.concierge.json.dumps", side_effect=TypeError("not serializable")):
            self.agent.on_summary({"key": "value"})
        captured = capsys.readouterr()
        assert "Error processing summary" in captured.out
        assert "not serializable" in captured.out


# ---------------------------------------------------------------------------
# Tool decorator metadata
# ---------------------------------------------------------------------------


class TestToolDecoratorMetadata:
    """Verify that @AgentBase.tool marks methods with expected metadata."""

    def test_check_availability_is_marked_as_tool(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        method = ConciergeAgent.check_availability
        assert getattr(method, "_is_tool", False) is True

    def test_check_availability_tool_name(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        assert ConciergeAgent.check_availability._tool_name == "check_availability"

    def test_check_availability_tool_params_has_description(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        params = ConciergeAgent.check_availability._tool_params
        assert "description" in params
        assert "parameters" in params
        assert "service" in params["parameters"]
        assert "date" in params["parameters"]
        assert "time" in params["parameters"]

    def test_get_directions_is_marked_as_tool(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        method = ConciergeAgent.get_directions
        assert getattr(method, "_is_tool", False) is True

    def test_get_directions_tool_name(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        assert ConciergeAgent.get_directions._tool_name == "get_directions"

    def test_get_directions_tool_params_has_location_parameter(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        params = ConciergeAgent.get_directions._tool_params
        assert "parameters" in params
        assert "location" in params["parameters"]


# ---------------------------------------------------------------------------
# Edge cases & configuration variants
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_empty_services_list(self):
        """Agent can be created with an empty services list."""
        agent, _ = _make_concierge(services=[])
        assert agent.services == []

    def test_empty_amenities_dict(self):
        """Agent can be created with an empty amenities dict."""
        agent, _ = _make_concierge(amenities={})
        assert agent.amenities == {}

    def test_amenity_with_many_fields(self):
        """Amenities can have arbitrary detail keys."""
        big_amenity = {
            "conference_room": {
                "hours": "8 AM - 10 PM",
                "location": "Mezzanine Level",
                "capacity": "200 guests",
                "av_equipment": "Projector, Microphone",
            }
        }
        agent, _ = _make_concierge(amenities=big_amenity)
        assert agent.amenities == big_amenity

    def test_check_availability_with_empty_services(self):
        """When services list is empty, every service is 'unknown'."""
        agent = _make_bare_concierge()
        agent.services = []
        result = agent.check_availability(
            {"service": "anything", "date": "2025-01-01", "time": "10:00"},
            raw_data={},
        )
        assert "sorry" in result.response.lower() or "don't offer" in result.response.lower()

    def test_get_directions_with_empty_amenities(self):
        """When amenities dict is empty, every location is unknown."""
        agent = _make_bare_concierge()
        agent.amenities = {}
        result = agent.get_directions({"location": "pool"}, raw_data={})
        assert "don't have" in result.response.lower() or "front desk" in result.response.lower()

    def test_hours_of_operation_none_gets_default(self):
        """Passing None for hours_of_operation uses the default."""
        agent, _ = _make_concierge(hours_of_operation=None)
        assert agent.hours_of_operation == {"default": "9 AM - 5 PM"}

    def test_special_instructions_none_gets_empty_list(self):
        """Passing None for special_instructions uses an empty list."""
        agent, _ = _make_concierge(special_instructions=None)
        assert agent.special_instructions == []


# ---------------------------------------------------------------------------
# Setup integration (prompt sections created in correct order)
# ---------------------------------------------------------------------------


class TestSetupSectionOrder:
    """Verify the order and count of prompt_add_section calls."""

    @pytest.fixture(autouse=True)
    def _create_agent(self):
        from signalwire_agents.prefabs.concierge import ConciergeAgent

        self.agent = ConciergeAgent.__new__(ConciergeAgent)
        self.agent.venue_name = VENUE_NAME
        self.agent.services = list(SERVICES)
        self.agent.amenities = dict(AMENITIES)
        self.agent.hours_of_operation = {"default": "9 AM - 5 PM"}
        self.agent.special_instructions = []
        self.agent.prompt_add_section = Mock()
        self.agent.set_post_prompt = Mock()
        self.agent.add_hints = Mock()
        self.agent.set_params = Mock()
        self.agent.set_global_data = Mock()
        self.agent.set_native_functions = Mock()

    def test_six_sections_created(self):
        """Exactly six prompt sections should be added."""
        self.agent._setup_concierge_agent()
        assert self.agent.prompt_add_section.call_count == 6

    def test_section_order(self):
        """Sections are created in the expected deterministic order."""
        self.agent._setup_concierge_agent()
        titles = [c[0][0] for c in self.agent.prompt_add_section.call_args_list]
        expected = [
            "Personality",
            "Goal",
            "Instructions",
            "Available Services",
            "Amenities",
            "Hours of Operation",
        ]
        assert titles == expected
