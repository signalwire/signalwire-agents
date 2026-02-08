"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for contexts module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from signalwire_agents.core.contexts import (
    ContextBuilder,
    Context,
    Step,
    create_simple_context
)


class TestStep:
    """Test Step functionality"""
    
    def test_basic_initialization(self):
        """Test basic Step initialization"""
        step = Step("greeting")
        
        assert step.name == "greeting"
        assert step._text is None
        assert step._step_criteria is None
        assert step._functions is None
        assert step._valid_steps is None
        assert step._sections == []
    
    def test_set_text(self):
        """Test setting step text"""
        step = Step("greeting")
        
        result = step.set_text("Hello, how can I help you today?")
        
        assert result is step  # Should return self for chaining
        assert step._text == "Hello, how can I help you today?"
    
    def test_add_section(self):
        """Test adding POM sections"""
        step = Step("greeting")
        
        result = step.add_section("Introduction", "Welcome to our service")
        
        assert result is step  # Should return self for chaining
        assert len(step._sections) == 1
        assert step._sections[0]["title"] == "Introduction"
        assert step._sections[0]["body"] == "Welcome to our service"
    
    def test_add_bullets(self):
        """Test adding bullet sections"""
        step = Step("greeting")
        bullets = ["First point", "Second point", "Third point"]
        
        result = step.add_bullets("Key Points", bullets)
        
        assert result is step  # Should return self for chaining
        assert len(step._sections) == 1
        assert step._sections[0]["title"] == "Key Points"
        assert step._sections[0]["bullets"] == bullets
    
    def test_set_step_criteria(self):
        """Test setting step criteria"""
        step = Step("greeting")
        
        result = step.set_step_criteria("User has provided their name")
        
        assert result is step  # Should return self for chaining
        assert step._step_criteria == "User has provided their name"
    
    def test_set_functions(self):
        """Test setting available functions"""
        step = Step("greeting")
        
        # Test with function list
        result = step.set_functions(["get_weather", "search"])
        assert result is step
        assert step._functions == ["get_weather", "search"]
        
        # Test with "none"
        step.set_functions("none")
        assert step._functions == "none"
    
    def test_set_valid_steps(self):
        """Test setting valid steps"""
        step = Step("greeting")
        valid_steps = ["next", "collect_info", "end"]
        
        result = step.set_valid_steps(valid_steps)
        
        assert result is step  # Should return self for chaining
        assert step._valid_steps == valid_steps
    
    def test_text_and_sections_conflict(self):
        """Test that text and sections cannot be mixed"""
        step = Step("greeting")
        
        # Set text first
        step.set_text("Hello")
        
        # Adding sections should raise error
        with pytest.raises(ValueError, match="Cannot add POM sections when set_text"):
            step.add_section("Title", "Body")
        
        with pytest.raises(ValueError, match="Cannot add POM sections when set_text"):
            step.add_bullets("Title", ["bullet"])
    
    def test_sections_and_text_conflict(self):
        """Test that sections and text cannot be mixed"""
        step = Step("greeting")
        
        # Add section first
        step.add_section("Title", "Body")
        
        # Setting text should raise error
        with pytest.raises(ValueError, match="Cannot use set_text\\(\\) when POM sections"):
            step.set_text("Hello")
    
    def test_render_text_with_text(self):
        """Test rendering text when text is set"""
        step = Step("greeting")
        step.set_text("Hello, how can I help you?")
        
        rendered = step._render_text()
        
        assert rendered == "Hello, how can I help you?"
    
    def test_render_text_with_sections(self):
        """Test rendering text from POM sections"""
        step = Step("greeting")
        step.add_section("Welcome", "Hello there!")
        step.add_bullets("Options", ["Option 1", "Option 2"])
        
        rendered = step._render_text()
        
        assert "## Welcome" in rendered
        assert "Hello there!" in rendered
        assert "## Options" in rendered
        assert "- Option 1" in rendered
        assert "- Option 2" in rendered
    
    def test_render_text_no_content(self):
        """Test rendering text when no content is set"""
        step = Step("greeting")
        
        with pytest.raises(ValueError, match="Step 'greeting' has no text or POM sections"):
            step._render_text()
    
    def test_to_dict_basic(self):
        """Test converting step to dictionary"""
        step = Step("greeting")
        step.set_text("Hello!")
        
        result = step.to_dict()
        
        assert result["text"] == "Hello!"
        assert "step_criteria" not in result
        assert "functions" not in result
        assert "valid_steps" not in result
    
    def test_to_dict_complete(self):
        """Test converting step with all fields to dictionary"""
        step = Step("greeting")
        step.set_text("Hello!")
        step.set_step_criteria("User responds")
        step.set_functions(["search"])
        step.set_valid_steps(["next"])
        
        result = step.to_dict()
        
        assert result["text"] == "Hello!"
        assert result["step_criteria"] == "User responds"
        assert result["functions"] == ["search"]
        assert result["valid_steps"] == ["next"]


class TestContext:
    """Test Context functionality"""
    
    def test_basic_initialization(self):
        """Test basic Context initialization"""
        context = Context("customer_service")
        
        assert context.name == "customer_service"
        assert context._steps == {}
        assert context._step_order == []
        assert context._valid_contexts is None
    
    def test_add_step(self):
        """Test adding steps to context"""
        context = Context("customer_service")
        
        step = context.add_step("greeting")
        
        assert isinstance(step, Step)
        assert step.name == "greeting"
        assert "greeting" in context._steps
        assert context._step_order == ["greeting"]
    
    def test_add_multiple_steps(self):
        """Test adding multiple steps"""
        context = Context("customer_service")
        
        step1 = context.add_step("greeting")
        step2 = context.add_step("collect_info")
        step3 = context.add_step("provide_solution")
        
        assert len(context._steps) == 3
        assert context._step_order == ["greeting", "collect_info", "provide_solution"]
        assert all(isinstance(step, Step) for step in [step1, step2, step3])
    
    def test_add_duplicate_step(self):
        """Test adding duplicate step names"""
        context = Context("customer_service")
        
        context.add_step("greeting")
        
        with pytest.raises(ValueError, match="Step 'greeting' already exists"):
            context.add_step("greeting")
    
    def test_set_valid_contexts(self):
        """Test setting valid contexts"""
        context = Context("customer_service")
        valid_contexts = ["sales", "technical_support"]
        
        result = context.set_valid_contexts(valid_contexts)
        
        assert result is context  # Should return self for chaining
        assert context._valid_contexts == valid_contexts
    
    def test_to_dict_basic(self):
        """Test converting context to dictionary"""
        context = Context("customer_service")
        step = context.add_step("greeting")
        step.set_text("Hello!")
        
        result = context.to_dict()
        
        assert "steps" in result
        assert len(result["steps"]) == 1
        assert result["steps"][0]["text"] == "Hello!"
        assert "valid_contexts" not in result
    
    def test_to_dict_with_valid_contexts(self):
        """Test converting context with valid contexts"""
        context = Context("customer_service")
        step = context.add_step("greeting")
        step.set_text("Hello!")
        context.set_valid_contexts(["sales"])
        
        result = context.to_dict()
        
        assert "steps" in result
        assert result["valid_contexts"] == ["sales"]
    
    def test_to_dict_no_steps(self):
        """Test converting context with no steps"""
        context = Context("customer_service")
        
        with pytest.raises(ValueError, match="Context 'customer_service' has no steps"):
            context.to_dict()


class TestContextBuilder:
    """Test ContextBuilder functionality"""
    
    def test_basic_initialization(self):
        """Test basic ContextBuilder initialization"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        # ContextBuilder doesn't store agent reference, just uses it during init
        assert builder._contexts == {}
    
    def test_add_context(self):
        """Test adding a context"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        context = builder.add_context("customer_service")
        assert isinstance(context, Context)
        assert "customer_service" in builder._contexts
    
    def test_add_duplicate_context(self):
        """Test adding duplicate context raises error"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        builder.add_context("customer_service")
        
        # The actual API raises an error for duplicates
        with pytest.raises(ValueError, match="Context 'customer_service' already exists"):
            builder.add_context("customer_service")
    
    def test_validate_success(self):
        """Test successful validation with default context"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        context = builder.add_context("default")  # Must be named 'default' for single context
        step = context.add_step("greeting")
        step.set_text("Hello!")
        
        # Should not raise any exceptions
        builder.validate()
    
    def test_validate_no_contexts(self):
        """Test validation with no contexts"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        with pytest.raises(ValueError, match="At least one context must be defined"):
            builder.validate()
    
    def test_validate_context_no_steps(self):
        """Test validation with context having no steps"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        builder.add_context("default")  # Must be named 'default' for single context
        
        with pytest.raises(ValueError, match="Context 'default' must have at least one step"):
            builder.validate()
    
    def test_to_dict(self):
        """Test converting builder to dictionary"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        context = builder.add_context("default")  # Must be named 'default' for single context
        step = context.add_step("greeting")
        step.set_text("Hello!")
        
        result = builder.to_dict()
        assert isinstance(result, dict)
        assert "default" in result


class TestCreateSimpleContext:
    """Test create_simple_context factory function"""
    
    def test_create_simple_context_default(self):
        """Test creating simple context with default name"""
        context = create_simple_context()
        
        assert isinstance(context, Context)
        assert context.name == "default"
    
    def test_create_simple_context_custom_name(self):
        """Test creating simple context with custom name"""
        context = create_simple_context("my_context")
        
        assert isinstance(context, Context)
        assert context.name == "my_context"


class TestContextIntegration:
    """Test context integration scenarios"""
    
    def test_complete_context_workflow(self):
        """Test complete context building workflow with multiple contexts"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        # Create customer service context
        customer_service = builder.add_context("customer_service")
        customer_service.set_valid_contexts(["sales", "technical_support"])
        
        # Add greeting step
        greeting = customer_service.add_step("greeting")
        greeting.set_text("Hello! Welcome to customer service. How can I help you today?")
        greeting.set_step_criteria("User has stated their issue")
        greeting.set_functions(["search_knowledge_base", "escalate_to_human"])
        greeting.set_valid_steps(["next", "gather_info"])  # Use valid step names
        
        # Add information gathering step
        gather_info = customer_service.add_step("gather_info")
        gather_info.add_section("Information Needed", "Please provide the following details:")
        gather_info.add_bullets("Required Information", [
            "Account number or phone number",
            "Description of the issue",
            "When did the issue start?"
        ])
        gather_info.set_step_criteria("All required information has been collected")
        gather_info.set_valid_steps(["next", "greeting"])
        
        # Add resolution step
        resolution = customer_service.add_step("resolution")
        resolution.set_text("Based on the information provided, here's how we can resolve your issue:")
        resolution.set_functions("none")  # No functions needed for final step
        
        # Add the referenced contexts to satisfy validation
        sales = builder.add_context("sales")
        sales_step = sales.add_step("sales_greeting")
        sales_step.set_text("Welcome to sales!")
        
        technical_support = builder.add_context("technical_support")
        tech_step = technical_support.add_step("tech_greeting")
        tech_step.set_text("Welcome to technical support!")
        
        # Validate the complete structure
        builder.validate()
        
        # Convert to dictionary
        result = builder.to_dict()
        assert "customer_service" in result
        assert "sales" in result
        assert "technical_support" in result
        assert len(result["customer_service"]["steps"]) == 3
    
    def test_multiple_contexts(self):
        """Test building multiple contexts"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        
        # Create sales context
        sales = builder.add_context("sales")
        sales_step = sales.add_step("pitch")
        sales_step.set_text("Let me tell you about our amazing products!")
        
        # Create support context
        support = builder.add_context("support")
        support_step = support.add_step("diagnose")
        support_step.set_text("Let's troubleshoot your issue.")
        
        # Validate and convert
        builder.validate()
        result = builder.to_dict()
        
        # Verify both contexts exist
        assert "sales" in result
        assert "support" in result
        assert len(result["sales"]["steps"]) == 1
        assert len(result["support"]["steps"]) == 1
    
    def test_complex_step_configuration(self):
        """Test complex step configuration with all features"""
        context = Context("complex")
        
        step = context.add_step("complex_step")
        
        # Use method chaining
        step.add_section("Overview", "This is a complex step with multiple sections") \
            .add_bullets("Features", ["Feature 1", "Feature 2", "Feature 3"]) \
            .add_section("Instructions", "Follow these steps carefully") \
            .set_step_criteria("All features have been demonstrated") \
            .set_functions(["demo_feature_1", "demo_feature_2", "demo_feature_3"]) \
            .set_valid_steps(["next", "previous", "help"])
        
        # Convert to dict and verify
        step_dict = step.to_dict()
        
        # Check that all sections are rendered
        text = step_dict["text"]
        assert "## Overview" in text
        assert "This is a complex step" in text
        assert "## Features" in text
        assert "- Feature 1" in text
        assert "## Instructions" in text
        assert "Follow these steps" in text
        
        # Check other fields
        assert step_dict["step_criteria"] == "All features have been demonstrated"
        assert step_dict["functions"] == ["demo_feature_1", "demo_feature_2", "demo_feature_3"]
        assert step_dict["valid_steps"] == ["next", "previous", "help"]


class TestStepResetAndContextNavigation:
    """Tests for Step reset parameters and context navigation features"""

    def test_set_valid_contexts(self):
        """Test setting valid contexts on a step"""
        step = Step("transfer")
        result = step.set_valid_contexts(["sales", "support"])
        assert result is step
        assert step._valid_contexts == ["sales", "support"]

    def test_set_reset_system_prompt(self):
        """Test setting reset system prompt"""
        step = Step("transfer")
        result = step.set_reset_system_prompt("You are a sales agent.")
        assert result is step
        assert step._reset_system_prompt == "You are a sales agent."

    def test_set_reset_user_prompt(self):
        """Test setting reset user prompt"""
        step = Step("transfer")
        result = step.set_reset_user_prompt("Please help me buy something.")
        assert result is step
        assert step._reset_user_prompt == "Please help me buy something."

    def test_set_reset_consolidate(self):
        """Test setting reset consolidate flag"""
        step = Step("transfer")
        result = step.set_reset_consolidate(True)
        assert result is step
        assert step._reset_consolidate is True

    def test_set_reset_full_reset(self):
        """Test setting reset full_reset flag"""
        step = Step("transfer")
        result = step.set_reset_full_reset(True)
        assert result is step
        assert step._reset_full_reset is True

    def test_to_dict_with_valid_contexts(self):
        """Test to_dict includes valid_contexts when set"""
        step = Step("transfer")
        step.set_text("Transferring you now.")
        step.set_valid_contexts(["sales", "support"])
        result = step.to_dict()
        assert result["valid_contexts"] == ["sales", "support"]

    def test_to_dict_with_partial_reset(self):
        """Test to_dict includes reset object with only system_prompt"""
        step = Step("transfer")
        step.set_text("Transferring you now.")
        step.set_reset_system_prompt("New system prompt")
        result = step.to_dict()
        assert "reset" in result
        assert result["reset"]["system_prompt"] == "New system prompt"
        assert "user_prompt" not in result["reset"]
        assert "consolidate" not in result["reset"]
        assert "full_reset" not in result["reset"]

    def test_to_dict_with_full_reset_object(self):
        """Test to_dict includes reset object with all fields"""
        step = Step("transfer")
        step.set_text("Transferring you now.")
        step.set_reset_system_prompt("New system prompt")
        step.set_reset_user_prompt("User message")
        step.set_reset_consolidate(True)
        step.set_reset_full_reset(True)
        result = step.to_dict()
        assert result["reset"]["system_prompt"] == "New system prompt"
        assert result["reset"]["user_prompt"] == "User message"
        assert result["reset"]["consolidate"] is True
        assert result["reset"]["full_reset"] is True

    def test_to_dict_no_reset_when_defaults(self):
        """Test to_dict omits reset when all reset fields are defaults"""
        step = Step("greeting")
        step.set_text("Hello!")
        result = step.to_dict()
        assert "reset" not in result


class TestContextEntryParameters:
    """Tests for Context entry parameters (system_prompt, consolidate, etc.)"""

    def test_set_post_prompt(self):
        """Test setting post prompt"""
        context = Context("sales")
        result = context.set_post_prompt("Summarize the conversation")
        assert result is context
        assert context._post_prompt == "Summarize the conversation"

    def test_set_system_prompt(self):
        """Test setting system prompt"""
        context = Context("sales")
        result = context.set_system_prompt("You are a sales agent.")
        assert result is context
        assert context._system_prompt == "You are a sales agent."

    def test_set_system_prompt_conflict_with_sections(self):
        """Test that set_system_prompt raises when sections already exist"""
        context = Context("sales")
        context.add_system_section("Role", "You are a sales agent.")
        with pytest.raises(ValueError, match="Cannot use set_system_prompt"):
            context.set_system_prompt("You are a sales agent.")

    def test_set_consolidate(self):
        """Test setting consolidate flag"""
        context = Context("sales")
        result = context.set_consolidate(True)
        assert result is context
        assert context._consolidate is True

    def test_set_full_reset(self):
        """Test setting full_reset flag"""
        context = Context("sales")
        result = context.set_full_reset(True)
        assert result is context
        assert context._full_reset is True

    def test_set_user_prompt(self):
        """Test setting user prompt"""
        context = Context("sales")
        result = context.set_user_prompt("I want to buy something.")
        assert result is context
        assert context._user_prompt == "I want to buy something."

    def test_set_isolated(self):
        """Test setting isolated flag"""
        context = Context("sales")
        result = context.set_isolated(True)
        assert result is context
        assert context._isolated is True


class TestContextSystemPromptSections:
    """Tests for Context system prompt POM sections"""

    def test_add_system_section(self):
        """Test adding a POM section to system prompt"""
        context = Context("sales")
        result = context.add_system_section("Role", "You are a sales agent.")
        assert result is context
        assert len(context._system_prompt_sections) == 1
        assert context._system_prompt_sections[0]["title"] == "Role"
        assert context._system_prompt_sections[0]["body"] == "You are a sales agent."

    def test_add_system_section_conflict_with_set_system_prompt(self):
        """Test that add_system_section raises when set_system_prompt already used"""
        context = Context("sales")
        context.set_system_prompt("You are a sales agent.")
        with pytest.raises(ValueError, match="Cannot add POM sections for system prompt"):
            context.add_system_section("Role", "You are a sales agent.")

    def test_add_system_bullets(self):
        """Test adding bullet points to system prompt"""
        context = Context("sales")
        result = context.add_system_bullets("Rules", ["Be polite", "Be helpful"])
        assert result is context
        assert len(context._system_prompt_sections) == 1
        assert context._system_prompt_sections[0]["bullets"] == ["Be polite", "Be helpful"]

    def test_add_system_bullets_conflict_with_set_system_prompt(self):
        """Test that add_system_bullets raises when set_system_prompt already used"""
        context = Context("sales")
        context.set_system_prompt("You are a sales agent.")
        with pytest.raises(ValueError, match="Cannot add POM sections for system prompt"):
            context.add_system_bullets("Rules", ["Be polite"])

    def test_render_system_prompt_with_text(self):
        """Test _render_system_prompt returns text when set"""
        context = Context("sales")
        context.set_system_prompt("You are a sales agent.")
        assert context._render_system_prompt() == "You are a sales agent."

    def test_render_system_prompt_with_sections(self):
        """Test _render_system_prompt renders POM sections"""
        context = Context("sales")
        context.add_system_section("Role", "You are a sales agent.")
        context.add_system_bullets("Rules", ["Be polite", "Be helpful"])
        rendered = context._render_system_prompt()
        assert "## Role" in rendered
        assert "You are a sales agent." in rendered
        assert "## Rules" in rendered
        assert "- Be polite" in rendered
        assert "- Be helpful" in rendered

    def test_render_system_prompt_none(self):
        """Test _render_system_prompt returns None when nothing is set"""
        context = Context("sales")
        assert context._render_system_prompt() is None


class TestContextPromptSections:
    """Tests for Context prompt (separate from system_prompt) POM sections"""

    def test_set_prompt(self):
        """Test setting context prompt text"""
        context = Context("sales")
        result = context.set_prompt("Welcome to the sales department.")
        assert result is context
        assert context._prompt_text == "Welcome to the sales department."

    def test_set_prompt_conflict_with_sections(self):
        """Test that set_prompt raises when sections already exist"""
        context = Context("sales")
        context.add_section("Greeting", "Welcome!")
        with pytest.raises(ValueError, match="Cannot use set_prompt"):
            context.set_prompt("Welcome!")

    def test_add_section(self):
        """Test adding a section to context prompt"""
        context = Context("sales")
        result = context.add_section("Greeting", "Welcome to our store!")
        assert result is context
        assert len(context._prompt_sections) == 1
        assert context._prompt_sections[0]["title"] == "Greeting"
        assert context._prompt_sections[0]["body"] == "Welcome to our store!"

    def test_add_section_conflict_with_set_prompt(self):
        """Test that add_section raises when set_prompt already used"""
        context = Context("sales")
        context.set_prompt("Welcome!")
        with pytest.raises(ValueError, match="Cannot add POM sections when set_prompt"):
            context.add_section("Greeting", "Welcome!")

    def test_add_bullets(self):
        """Test adding bullet points to context prompt"""
        context = Context("sales")
        result = context.add_bullets("Products", ["Widget A", "Widget B"])
        assert result is context
        assert len(context._prompt_sections) == 1
        assert context._prompt_sections[0]["bullets"] == ["Widget A", "Widget B"]

    def test_add_bullets_conflict_with_set_prompt(self):
        """Test that add_bullets raises when set_prompt already used"""
        context = Context("sales")
        context.set_prompt("Welcome!")
        with pytest.raises(ValueError, match="Cannot add POM sections when set_prompt"):
            context.add_bullets("Products", ["Widget A"])

    def test_render_prompt_with_text(self):
        """Test _render_prompt returns text when set"""
        context = Context("sales")
        context.set_prompt("Welcome to sales.")
        assert context._render_prompt() == "Welcome to sales."

    def test_render_prompt_with_sections(self):
        """Test _render_prompt renders POM sections"""
        context = Context("sales")
        context.add_section("Greeting", "Hello!")
        context.add_bullets("Items", ["Item 1", "Item 2"])
        rendered = context._render_prompt()
        assert "## Greeting" in rendered
        assert "Hello!" in rendered
        assert "## Items" in rendered
        assert "- Item 1" in rendered
        assert "- Item 2" in rendered

    def test_render_prompt_none(self):
        """Test _render_prompt returns None when nothing is set"""
        context = Context("sales")
        assert context._render_prompt() is None


class TestContextFillers:
    """Tests for Context enter/exit filler functionality"""

    def test_set_enter_fillers(self):
        """Test setting enter fillers"""
        context = Context("sales")
        fillers = {"en-US": ["Welcome!", "Hello!"], "default": ["Hi!"]}
        result = context.set_enter_fillers(fillers)
        assert result is context
        assert context._enter_fillers == fillers

    def test_set_enter_fillers_empty_dict(self):
        """Test that empty dict does not set enter fillers"""
        context = Context("sales")
        result = context.set_enter_fillers({})
        assert result is context
        assert context._enter_fillers is None

    def test_set_enter_fillers_non_dict(self):
        """Test that non-dict does not set enter fillers"""
        context = Context("sales")
        result = context.set_enter_fillers(None)
        assert result is context
        assert context._enter_fillers is None

    def test_set_exit_fillers(self):
        """Test setting exit fillers"""
        context = Context("sales")
        fillers = {"en-US": ["Goodbye!", "Thank you!"], "default": ["Bye!"]}
        result = context.set_exit_fillers(fillers)
        assert result is context
        assert context._exit_fillers == fillers

    def test_set_exit_fillers_empty_dict(self):
        """Test that empty dict does not set exit fillers"""
        context = Context("sales")
        result = context.set_exit_fillers({})
        assert result is context
        assert context._exit_fillers is None

    def test_set_exit_fillers_non_dict(self):
        """Test that non-dict does not set exit fillers"""
        context = Context("sales")
        result = context.set_exit_fillers(None)
        assert result is context
        assert context._exit_fillers is None

    def test_add_enter_filler(self):
        """Test adding enter fillers for a specific language"""
        context = Context("sales")
        result = context.add_enter_filler("en-US", ["Welcome!", "Hello!"])
        assert result is context
        assert context._enter_fillers == {"en-US": ["Welcome!", "Hello!"]}

    def test_add_enter_filler_multiple_languages(self):
        """Test adding enter fillers for multiple languages"""
        context = Context("sales")
        context.add_enter_filler("en-US", ["Welcome!"])
        context.add_enter_filler("es", ["Bienvenido!"])
        assert context._enter_fillers == {"en-US": ["Welcome!"], "es": ["Bienvenido!"]}

    def test_add_enter_filler_invalid_inputs(self):
        """Test that invalid inputs to add_enter_filler are ignored"""
        context = Context("sales")
        context.add_enter_filler("", ["Hello!"])
        assert context._enter_fillers is None
        context.add_enter_filler("en-US", [])
        assert context._enter_fillers is None
        context.add_enter_filler("en-US", None)
        assert context._enter_fillers is None

    def test_add_exit_filler(self):
        """Test adding exit fillers for a specific language"""
        context = Context("sales")
        result = context.add_exit_filler("en-US", ["Goodbye!", "See you!"])
        assert result is context
        assert context._exit_fillers == {"en-US": ["Goodbye!", "See you!"]}

    def test_add_exit_filler_multiple_languages(self):
        """Test adding exit fillers for multiple languages"""
        context = Context("sales")
        context.add_exit_filler("en-US", ["Goodbye!"])
        context.add_exit_filler("es", ["Adios!"])
        assert context._exit_fillers == {"en-US": ["Goodbye!"], "es": ["Adios!"]}

    def test_add_exit_filler_invalid_inputs(self):
        """Test that invalid inputs to add_exit_filler are ignored"""
        context = Context("sales")
        context.add_exit_filler("", ["Bye!"])
        assert context._exit_fillers is None
        context.add_exit_filler("en-US", [])
        assert context._exit_fillers is None
        context.add_exit_filler("en-US", None)
        assert context._exit_fillers is None


class TestContextToDictComprehensive:
    """Tests for Context.to_dict with various combinations of parameters"""

    def test_to_dict_with_post_prompt(self):
        """Test to_dict includes post_prompt"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_post_prompt("Summarize the conversation")
        result = context.to_dict()
        assert result["post_prompt"] == "Summarize the conversation"

    def test_to_dict_with_system_prompt(self):
        """Test to_dict includes system_prompt"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_system_prompt("You are a sales agent.")
        result = context.to_dict()
        assert result["system_prompt"] == "You are a sales agent."

    def test_to_dict_with_system_sections(self):
        """Test to_dict renders system prompt from POM sections"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.add_system_section("Role", "You are a sales agent.")
        result = context.to_dict()
        assert "system_prompt" in result
        assert "## Role" in result["system_prompt"]

    def test_to_dict_with_consolidate_and_full_reset(self):
        """Test to_dict includes consolidate and full_reset"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_consolidate(True)
        context.set_full_reset(True)
        result = context.to_dict()
        assert result["consolidate"] is True
        assert result["full_reset"] is True

    def test_to_dict_with_user_prompt(self):
        """Test to_dict includes user_prompt"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_user_prompt("I want to buy something.")
        result = context.to_dict()
        assert result["user_prompt"] == "I want to buy something."

    def test_to_dict_with_isolated(self):
        """Test to_dict includes isolated"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_isolated(True)
        result = context.to_dict()
        assert result["isolated"] is True

    def test_to_dict_with_prompt_text(self):
        """Test to_dict includes prompt when set_prompt used"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_prompt("Welcome to the sales department.")
        result = context.to_dict()
        assert result["prompt"] == "Welcome to the sales department."
        assert "pom" not in result

    def test_to_dict_with_prompt_sections(self):
        """Test to_dict includes pom when sections added"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.add_section("Greeting", "Welcome!")
        context.add_bullets("Products", ["Widget A", "Widget B"])
        result = context.to_dict()
        assert "pom" in result
        assert "prompt" not in result
        assert len(result["pom"]) == 2

    def test_to_dict_with_enter_and_exit_fillers(self):
        """Test to_dict includes fillers"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        context.set_enter_fillers({"en-US": ["Welcome!"]})
        context.set_exit_fillers({"en-US": ["Goodbye!"]})
        result = context.to_dict()
        assert result["enter_fillers"] == {"en-US": ["Welcome!"]}
        assert result["exit_fillers"] == {"en-US": ["Goodbye!"]}

    def test_to_dict_omits_defaults(self):
        """Test to_dict omits fields that are at default values"""
        context = Context("sales")
        context.add_step("greeting").set_text("Hello!")
        result = context.to_dict()
        assert "post_prompt" not in result
        assert "system_prompt" not in result
        assert "consolidate" not in result
        assert "full_reset" not in result
        assert "user_prompt" not in result
        assert "isolated" not in result
        assert "prompt" not in result
        assert "pom" not in result
        assert "enter_fillers" not in result
        assert "exit_fillers" not in result


class TestContextBuilderValidation:
    """Tests for ContextBuilder validation edge cases"""

    def test_validate_single_context_not_named_default(self):
        """Test validation fails when single context is not named 'default'"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        context = builder.add_context("custom_name")
        context.add_step("greeting").set_text("Hello!")
        with pytest.raises(ValueError, match="single context, it must be named 'default'"):
            builder.validate()

    def test_validate_invalid_step_reference(self):
        """Test validation fails when valid_steps references unknown step"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        context = builder.add_context("default")
        step = context.add_step("greeting")
        step.set_text("Hello!")
        step.set_valid_steps(["nonexistent_step"])
        with pytest.raises(ValueError, match="references unknown step 'nonexistent_step'"):
            builder.validate()

    def test_validate_next_is_allowed_in_valid_steps(self):
        """Test validation allows 'next' as a valid step reference"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        context = builder.add_context("default")
        step = context.add_step("greeting")
        step.set_text("Hello!")
        step.set_valid_steps(["next"])
        builder.validate()  # Should not raise

    def test_validate_invalid_context_reference_at_context_level(self):
        """Test validation fails for unknown context in context-level valid_contexts"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        ctx1 = builder.add_context("ctx1")
        ctx1.add_step("s1").set_text("Hello!")
        ctx1.set_valid_contexts(["nonexistent_context"])
        ctx2 = builder.add_context("ctx2")
        ctx2.add_step("s2").set_text("Hi!")
        with pytest.raises(ValueError, match="Context 'ctx1' references unknown context 'nonexistent_context'"):
            builder.validate()

    def test_validate_invalid_context_reference_at_step_level(self):
        """Test validation fails for unknown context in step-level valid_contexts"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        ctx1 = builder.add_context("ctx1")
        step = ctx1.add_step("s1")
        step.set_text("Hello!")
        step.set_valid_contexts(["nonexistent_context"])
        ctx2 = builder.add_context("ctx2")
        ctx2.add_step("s2").set_text("Hi!")
        with pytest.raises(ValueError, match="references unknown context 'nonexistent_context'"):
            builder.validate()

    def test_validate_valid_context_references(self):
        """Test validation passes with valid context references at both levels"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        ctx1 = builder.add_context("ctx1")
        step1 = ctx1.add_step("s1")
        step1.set_text("Hello!")
        step1.set_valid_contexts(["ctx2"])
        ctx1.set_valid_contexts(["ctx2"])
        ctx2 = builder.add_context("ctx2")
        ctx2.add_step("s2").set_text("Hi!")
        builder.validate()  # Should not raise

    def test_to_dict_preserves_order(self):
        """Test that to_dict preserves context insertion order"""
        mock_agent = Mock()
        builder = ContextBuilder(mock_agent)
        ctx1 = builder.add_context("alpha")
        ctx1.add_step("s1").set_text("A")
        ctx2 = builder.add_context("beta")
        ctx2.add_step("s2").set_text("B")
        ctx3 = builder.add_context("gamma")
        ctx3.add_step("s3").set_text("C")
        result = builder.to_dict()
        assert list(result.keys()) == ["alpha", "beta", "gamma"]