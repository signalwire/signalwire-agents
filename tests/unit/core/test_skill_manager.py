"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SkillManager class
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from signalwire_agents.core.skill_manager import SkillManager
from signalwire_agents.core.skill_base import SkillBase


class MockSkill(SkillBase):
    """Mock skill for testing"""
    SKILL_NAME = "mock_skill"
    SKILL_DESCRIPTION = "A mock skill for testing"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []
    REQUIRED_ENV_VARS = []
    SUPPORTS_MULTIPLE_INSTANCES = False

    @classmethod
    def get_parameter_schema(cls):
        schema = super().get_parameter_schema()
        schema["mock_param"] = {
            "type": "string",
            "description": "A mock parameter",
            "default": "default_value",
            "required": False
        }
        return schema

    def __init__(self, agent, params=None):
        super().__init__(agent, params)
        self.setup_called = False
        self.register_tools_called = False
        self.cleanup_called = False

    def setup(self):
        self.setup_called = True
        return True

    def register_tools(self):
        self.register_tools_called = True
        self.agent.define_tool(
            name="mock_tool",
            description="A mock tool",
            parameters={"type": "object", "properties": {}},
            handler=lambda: {"result": "mock"}
        )


class FailingMockSkill(SkillBase):
    """Mock skill that fails setup"""
    SKILL_NAME = "failing_skill"
    SKILL_DESCRIPTION = "A skill that fails setup"
    SKILL_VERSION = "1.0.0"
    SUPPORTS_MULTIPLE_INSTANCES = False

    @classmethod
    def get_parameter_schema(cls):
        schema = super().get_parameter_schema()
        schema["fail_param"] = {
            "type": "string",
            "description": "A fail parameter",
            "required": False
        }
        return schema

    def setup(self):
        return False

    def register_tools(self):
        pass


class TestSkillManagerBasic:
    """Test basic SkillManager functionality"""
    
    def test_initialization(self, mock_agent):
        """Test SkillManager initialization"""
        skill_manager = SkillManager(mock_agent)
        
        assert skill_manager.agent is mock_agent
        assert skill_manager.loaded_skills == {}
    
    def test_agent_reference(self, mock_agent):
        """Test that skill manager maintains agent reference"""
        skill_manager = SkillManager(mock_agent)
        
        assert skill_manager.agent is mock_agent


class TestSkillManagerLoading:
    """Test skill loading functionality"""
    
    def test_load_skill_success(self, mock_agent):
        """Test successful skill loading"""
        skill_manager = SkillManager(mock_agent)
        
        success, error = skill_manager.load_skill("mock_skill", MockSkill)
        
        assert success is True
        assert error == ""
        assert len(skill_manager.loaded_skills) == 1
        
        # Check that skill was properly initialized
        skill_instance = list(skill_manager.loaded_skills.values())[0]
        assert isinstance(skill_instance, MockSkill)
        assert skill_instance.setup_called is True
        assert skill_instance.register_tools_called is True
    
    def test_load_skill_with_params(self, mock_agent):
        """Test loading skill with parameters"""
        skill_manager = SkillManager(mock_agent)
        
        params = {"param1": "value1", "param2": "value2"}
        success, error = skill_manager.load_skill("mock_skill", MockSkill, params=params)
        
        assert success is True
        skill_instance = list(skill_manager.loaded_skills.values())[0]
        assert skill_instance.params == params
    
    def test_load_skill_setup_failure(self, mock_agent):
        """Test loading skill that fails setup"""
        skill_manager = SkillManager(mock_agent)
        
        success, error = skill_manager.load_skill("failing_skill", FailingMockSkill)
        
        assert success is False
        assert "Failed to setup skill" in error
        assert len(skill_manager.loaded_skills) == 0
    
    def test_load_already_loaded_skill(self, mock_agent):
        """Test loading skill that's already loaded"""
        skill_manager = SkillManager(mock_agent)
        
        # Load first time
        success1, error1 = skill_manager.load_skill("mock_skill", MockSkill)
        assert success1 is True
        
        # Load second time - should fail for single-instance skills
        success2, error2 = skill_manager.load_skill("mock_skill", MockSkill)
        assert success2 is False
        assert "already loaded" in error2
    
    def test_load_skill_initialization_error(self, mock_agent):
        """Test loading skill that fails during initialization"""
        skill_manager = SkillManager(mock_agent)
        
        class BrokenSkill(SkillBase):
            SKILL_NAME = "broken_skill"
            SKILL_DESCRIPTION = "A broken skill"
            SUPPORTS_MULTIPLE_INSTANCES = False

            @classmethod
            def get_parameter_schema(cls):
                schema = super().get_parameter_schema()
                schema["broken_param"] = {"type": "string", "description": "A param", "required": False}
                return schema

            def __init__(self, agent, params=None):
                raise Exception("Initialization failed")

            def setup(self):
                return True

            def register_tools(self):
                pass
        
        success, error = skill_manager.load_skill("broken_skill", BrokenSkill)
        
        assert success is False
        assert "Error loading skill" in error
    
    def test_load_skill_without_class_registry_missing(self, mock_agent):
        """Test loading skill without providing class when registry is missing"""
        skill_manager = SkillManager(mock_agent)
        
        with patch('signalwire_agents.skills.registry.skill_registry') as mock_registry:
            mock_registry.get_skill_class.return_value = None
            
            success, error = skill_manager.load_skill("nonexistent_skill")
            
            assert success is False
            assert "not found in registry" in error


class TestSkillManagerUnloading:
    """Test skill unloading functionality"""
    
    def test_unload_skill_success(self, mock_agent):
        """Test successful skill unloading"""
        skill_manager = SkillManager(mock_agent)
        
        # Load skill first
        skill_manager.load_skill("mock_skill", MockSkill)
        assert len(skill_manager.loaded_skills) == 1
        
        # Get the instance key
        instance_key = list(skill_manager.loaded_skills.keys())[0]
        
        # Unload skill
        success = skill_manager.unload_skill(instance_key)
        
        assert success is True
        assert len(skill_manager.loaded_skills) == 0
    
    def test_unload_nonexistent_skill(self, mock_agent):
        """Test unloading non-existent skill"""
        skill_manager = SkillManager(mock_agent)
        
        success = skill_manager.unload_skill("nonexistent_skill")
        
        assert success is False
    
    def test_unload_skill_cleanup_called(self, mock_agent):
        """Test that cleanup is called during unloading"""
        skill_manager = SkillManager(mock_agent)
        
        class CleanupSkill(MockSkill):
            SKILL_NAME = "cleanup_skill"
            
            def cleanup(self):
                self.cleanup_called = True
        
        # Load and unload skill
        skill_manager.load_skill("cleanup_skill", CleanupSkill)
        skill_instance = list(skill_manager.loaded_skills.values())[0]
        instance_key = list(skill_manager.loaded_skills.keys())[0]
        
        skill_manager.unload_skill(instance_key)
        
        assert skill_instance.cleanup_called is True


class TestSkillManagerQueries:
    """Test skill query functionality"""
    
    def test_list_loaded_skills(self, mock_agent):
        """Test listing loaded skills"""
        skill_manager = SkillManager(mock_agent)
        
        # Initially empty
        loaded = skill_manager.list_loaded_skills()
        assert len(loaded) == 0
        
        # Load skill - only one will load due to single instance restriction
        skill_manager.load_skill("skill1", MockSkill)
        # This will fail because MockSkill doesn't support multiple instances
        skill_manager.load_skill("skill2", MockSkill)
        
        loaded = skill_manager.list_loaded_skills()
        # Only one skill should be loaded due to single instance restriction
        assert len(loaded) == 1
    
    def test_has_skill_loaded(self, mock_agent):
        """Test checking if skill is loaded"""
        skill_manager = SkillManager(mock_agent)
        
        # Not loaded initially
        assert skill_manager.has_skill("mock_skill") is False
        
        # Load skill
        skill_manager.load_skill("mock_skill", MockSkill)
        assert skill_manager.has_skill("mock_skill") is True
        
        # Unload skill
        instance_key = list(skill_manager.loaded_skills.keys())[0]
        skill_manager.unload_skill(instance_key)
        assert skill_manager.has_skill("mock_skill") is False
    
    def test_has_skill_nonexistent(self, mock_agent):
        """Test checking for non-existent skill"""
        skill_manager = SkillManager(mock_agent)
        
        assert skill_manager.has_skill("nonexistent_skill") is False
    
    def test_get_skill_instance(self, mock_agent):
        """Test getting skill instance"""
        skill_manager = SkillManager(mock_agent)
        
        # Load skill
        skill_manager.load_skill("mock_skill", MockSkill)
        
        # Get instance by skill name
        instance = skill_manager.get_skill("mock_skill")
        assert isinstance(instance, MockSkill)
        assert instance.agent is mock_agent
    
    def test_get_skill_instance_not_loaded(self, mock_agent):
        """Test getting instance of non-loaded skill"""
        skill_manager = SkillManager(mock_agent)
        
        instance = skill_manager.get_skill("nonexistent_skill")
        assert instance is None


class TestSkillManagerValidation:
    """Test skill validation functionality"""
    
    def test_validate_skill_requirements_success(self, mock_agent):
        """Test successful skill requirement validation"""
        skill_manager = SkillManager(mock_agent)
        
        class ValidSkill(MockSkill):
            SKILL_NAME = "valid_skill"
            REQUIRED_PACKAGES = []  # No requirements
            REQUIRED_ENV_VARS = []
        
        success, error = skill_manager.load_skill("valid_skill", ValidSkill)
        assert success is True
    
    def test_validate_skill_missing_env_vars(self, mock_agent):
        """Test skill with missing environment variables"""
        skill_manager = SkillManager(mock_agent)
        
        class EnvSkill(MockSkill):
            SKILL_NAME = "env_skill"
            REQUIRED_ENV_VARS = ["MISSING_ENV_VAR"]
        
        success, error = skill_manager.load_skill("env_skill", EnvSkill)
        assert success is False
        assert "Missing required environment variables" in error
    
    def test_validate_skill_with_env_vars(self, mock_agent, mock_env_vars):
        """Test skill with required environment variables present"""
        skill_manager = SkillManager(mock_agent)
        
        class EnvSkill(MockSkill):
            SKILL_NAME = "env_skill"
            REQUIRED_ENV_VARS = ["SIGNALWIRE_PROJECT_ID"]  # This is in mock_env_vars
        
        success, error = skill_manager.load_skill("env_skill", EnvSkill)
        assert success is True
    
    def test_validate_skill_missing_packages(self, mock_agent):
        """Test skill with missing packages"""
        skill_manager = SkillManager(mock_agent)
        
        class PackageSkill(MockSkill):
            SKILL_NAME = "package_skill"
            REQUIRED_PACKAGES = ["nonexistent_package_xyz"]
        
        success, error = skill_manager.load_skill("package_skill", PackageSkill)
        assert success is False
        assert "Missing required packages" in error


class TestSkillManagerErrorHandling:
    """Test error handling and edge cases"""
    
    def test_load_skill_exception_during_setup(self, mock_agent):
        """Test loading skill that raises exception during setup"""
        skill_manager = SkillManager(mock_agent)
        
        class ExceptionSkill(MockSkill):
            SKILL_NAME = "exception_skill"
            
            def setup(self):
                raise Exception("Setup failed")
        
        success, error = skill_manager.load_skill("exception_skill", ExceptionSkill)
        
        assert success is False
        assert "Error loading skill" in error
    
    def test_load_skill_exception_during_register_tools(self, mock_agent):
        """Test loading skill that raises exception during tool registration"""
        skill_manager = SkillManager(mock_agent)
        
        class ExceptionSkill(MockSkill):
            SKILL_NAME = "exception_skill"
            
            def register_tools(self):
                raise Exception("Tool registration failed")
        
        success, error = skill_manager.load_skill("exception_skill", ExceptionSkill)
        
        assert success is False
        assert "Error loading skill" in error


class TestSkillManagerIntegration:
    """Test integration with other components"""
    
    def test_skill_tool_registration_with_agent(self, mock_agent):
        """Test that skill tools are properly registered with agent"""
        skill_manager = SkillManager(mock_agent)
        
        # Mock the agent's define_tool method
        mock_agent.define_tool = Mock()
        
        success, error = skill_manager.load_skill("mock_skill", MockSkill)
        
        assert success is True
        # Should have called agent.define_tool
        mock_agent.define_tool.assert_called_once()
    
    def test_multiple_skills_loaded(self, mock_agent):
        """Test loading multiple skills with different names"""
        skill_manager = SkillManager(mock_agent)
        
        class Skill1(MockSkill):
            SKILL_NAME = "skill1"
            
            def register_tools(self):
                self.register_tools_called = True
                self.agent.define_tool(
                    name="skill1_tool",
                    description="A skill1 tool",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda: {"result": "skill1"}
                )
        
        class Skill2(MockSkill):
            SKILL_NAME = "skill2"
            
            def register_tools(self):
                self.register_tools_called = True
                self.agent.define_tool(
                    name="skill2_tool",
                    description="A skill2 tool",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda: {"result": "skill2"}
                )
        
        # Load both skills - should work since they have different names and tools
        success1, _ = skill_manager.load_skill("skill1", Skill1)
        success2, _ = skill_manager.load_skill("skill2", Skill2)
        
        assert success1 is True
        assert success2 is True
        assert len(skill_manager.list_loaded_skills()) == 2
    
    def test_skill_unload_cleanup_order(self, mock_agent):
        """Test that skills are cleaned up in proper order"""
        skill_manager = SkillManager(mock_agent)
        
        cleanup_order = []
        
        class OrderedSkill(MockSkill):
            def __init__(self, agent, params=None, skill_id=None):
                super().__init__(agent, params)
                self.skill_id = skill_id
            
            def cleanup(self):
                cleanup_order.append(self.skill_id)
        
        # Create skill classes with different IDs and names
        class Skill1(OrderedSkill):
            SKILL_NAME = "skill1"
            def __init__(self, agent, params=None):
                super().__init__(agent, params, "skill1")
                
            def register_tools(self):
                self.register_tools_called = True
                self.agent.define_tool(
                    name="skill1_tool",
                    description="A skill1 tool",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda: {"result": "skill1"}
                )
        
        class Skill2(OrderedSkill):
            SKILL_NAME = "skill2"
            def __init__(self, agent, params=None):
                super().__init__(agent, params, "skill2")
                
            def register_tools(self):
                self.register_tools_called = True
                self.agent.define_tool(
                    name="skill2_tool",
                    description="A skill2 tool",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda: {"result": "skill2"}
                )
        
        # Load skills
        skill_manager.load_skill("skill1", Skill1)
        skill_manager.load_skill("skill2", Skill2)
        
        # Get instance keys
        instance_keys = list(skill_manager.loaded_skills.keys())
        
        # Should have 2 skills loaded
        assert len(instance_keys) == 2
        
        # Unload in different order
        skill_manager.unload_skill(instance_keys[1])
        skill_manager.unload_skill(instance_keys[0])
        
        assert len(cleanup_order) == 2 