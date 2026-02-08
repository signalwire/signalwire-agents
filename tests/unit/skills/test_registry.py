"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for skills registry module
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from signalwire_agents.skills.registry import SkillRegistry, skill_registry
from signalwire_agents.core.skill_base import SkillBase


class MockSkill(SkillBase):
    """Mock skill for testing"""
    SKILL_NAME = "mock_skill"
    SKILL_DESCRIPTION = "A mock skill for testing"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]
    REQUIRED_ENV_VARS = ["API_KEY"]
    SUPPORTS_MULTIPLE_INSTANCES = True

    @classmethod
    def get_parameter_schema(cls):
        schema = super().get_parameter_schema()
        schema["test_param"] = {"type": "string", "description": "test", "required": False}
        return schema

    def setup(self):
        pass

    def register_tools(self):
        pass


class AnotherMockSkill(SkillBase):
    """Another mock skill for testing"""
    SKILL_NAME = "another_mock_skill"
    SKILL_DESCRIPTION = "Another mock skill"
    SKILL_VERSION = "2.0.0"
    REQUIRED_PACKAGES = []
    REQUIRED_ENV_VARS = []
    SUPPORTS_MULTIPLE_INSTANCES = False

    @classmethod
    def get_parameter_schema(cls):
        schema = super().get_parameter_schema()
        schema["test_param"] = {"type": "string", "description": "test", "required": False}
        return schema

    def setup(self):
        pass

    def register_tools(self):
        pass


class InvalidSkill(SkillBase):
    """Invalid skill without SKILL_NAME"""
    SKILL_NAME = None
    
    def setup(self):
        pass
    
    def register_tools(self):
        pass


class TestSkillRegistry:
    """Test SkillRegistry functionality"""
    
    def test_basic_initialization(self):
        """Test basic SkillRegistry initialization"""
        registry = SkillRegistry()

        assert registry._skills == {}
        assert registry._entry_points_loaded is False
        assert registry.logger is not None
    
    def test_register_skill_basic(self):
        """Test basic skill registration"""
        registry = SkillRegistry()
        
        registry.register_skill(MockSkill)
        
        assert "mock_skill" in registry._skills
        assert registry._skills["mock_skill"] == MockSkill
    
    def test_register_skill_duplicate(self):
        """Test registering duplicate skill"""
        registry = SkillRegistry()
        
        registry.register_skill(MockSkill)
        
        # Register the same skill again
        with patch.object(registry.logger, 'warning') as mock_warning:
            registry.register_skill(MockSkill)
            mock_warning.assert_called_once_with("Skill 'mock_skill' already registered")
        
        # Should still only have one instance
        assert len(registry._skills) == 1
    
    def test_register_multiple_skills(self):
        """Test registering multiple skills"""
        registry = SkillRegistry()
        
        registry.register_skill(MockSkill)
        registry.register_skill(AnotherMockSkill)
        
        assert len(registry._skills) == 2
        assert "mock_skill" in registry._skills
        assert "another_mock_skill" in registry._skills
    
    def test_get_skill_class_existing(self):
        """Test getting existing skill class"""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)

        skill_class = registry.get_skill_class("mock_skill")

        assert skill_class == MockSkill
    
    def test_get_skill_class_nonexistent(self):
        """Test getting nonexistent skill class"""
        registry = SkillRegistry()

        # Mock on-demand loading to prevent real filesystem scanning
        with patch.object(registry, '_load_skill_on_demand', return_value=None):
            skill_class = registry.get_skill_class("nonexistent_skill")

        assert skill_class is None
    
    @patch.object(SkillRegistry, '_load_skill_on_demand', return_value=None)
    def test_get_skill_class_triggers_on_demand_loading(self, mock_load):
        """Test that get_skill_class triggers on-demand loading for unknown skills"""
        registry = SkillRegistry()

        registry.get_skill_class("some_skill")

        mock_load.assert_called_once_with("some_skill")
    
    def test_list_skills_empty(self):
        """Test listing skills when no skill directories exist"""
        registry = SkillRegistry()

        # Mock the skills directory to return no subdirectories
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []
        with patch('signalwire_agents.skills.registry.Path') as mock_path_cls:
            mock_path_cls.return_value.parent = mock_skills_dir
            skills = registry.list_skills()

        assert skills == []
    
    def test_list_skills_with_skills(self):
        """Test listing skills with registered skills"""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)
        registry.register_skill(AnotherMockSkill)

        # Mock the filesystem to simulate two skill directories
        mock_dir1 = Mock()
        mock_dir1.is_dir.return_value = True
        mock_dir1.name = "mock_skill"
        mock_skill_file1 = Mock()
        mock_skill_file1.exists.return_value = True
        mock_dir1.__truediv__ = Mock(return_value=mock_skill_file1)

        mock_dir2 = Mock()
        mock_dir2.is_dir.return_value = True
        mock_dir2.name = "another_mock_skill"
        mock_skill_file2 = Mock()
        mock_skill_file2.exists.return_value = True
        mock_dir2.__truediv__ = Mock(return_value=mock_skill_file2)

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = [mock_dir1, mock_dir2]

        with patch('signalwire_agents.skills.registry.Path') as mock_path_cls:
            mock_path_cls.return_value.parent = mock_skills_dir
            skills = registry.list_skills()

        assert len(skills) == 2

        # Check first skill
        mock_skill_info = next(s for s in skills if s["name"] == "mock_skill")
        assert mock_skill_info["description"] == "A mock skill for testing"
        assert mock_skill_info["version"] == "1.0.0"
        assert mock_skill_info["required_packages"] == ["requests"]
        assert mock_skill_info["required_env_vars"] == ["API_KEY"]
        assert mock_skill_info["supports_multiple_instances"] is True

        # Check second skill
        another_skill_info = next(s for s in skills if s["name"] == "another_mock_skill")
        assert another_skill_info["description"] == "Another mock skill"
        assert another_skill_info["version"] == "2.0.0"
        assert another_skill_info["required_packages"] == []
        assert another_skill_info["required_env_vars"] == []
        assert another_skill_info["supports_multiple_instances"] is False
    
    def test_list_skills_triggers_on_demand_loading(self):
        """Test that list_skills triggers on-demand loading for found skill directories"""
        registry = SkillRegistry()

        # Mock a skill directory on the filesystem
        mock_dir = Mock()
        mock_dir.is_dir.return_value = True
        mock_dir.name = "some_skill"
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = True
        mock_dir.__truediv__ = Mock(return_value=mock_skill_file)

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = [mock_dir]

        with patch('signalwire_agents.skills.registry.Path') as mock_path_cls:
            mock_path_cls.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_skill_on_demand', return_value=None) as mock_load:
                registry.list_skills()
                mock_load.assert_called_once_with("some_skill")


class TestSkillDiscovery:
    """Test skill discovery functionality (on-demand loading)"""

    def test_discover_skills_is_noop(self):
        """Test that discover_skills is a no-op for backwards compatibility"""
        registry = SkillRegistry()

        # Should not raise and should not change state
        registry.discover_skills()
        assert registry._skills == {}

    def test_entry_points_loaded_idempotent(self):
        """Test that _load_entry_points is idempotent"""
        registry = SkillRegistry()

        mock_eps = MagicMock()
        mock_eps.return_value = MagicMock(select=MagicMock(return_value=[]))
        with patch('importlib.metadata.entry_points', mock_eps):
            registry._load_entry_points()
            registry._load_entry_points()  # Call again

            # Should only call entry_points once due to _entry_points_loaded flag
            assert mock_eps.call_count == 1

    def test_list_skills_scans_directory(self):
        """Test that list_skills scans the skills directory"""
        registry = SkillRegistry()

        # Mock the skills directory structure
        mock_skill_dir1 = Mock()
        mock_skill_dir1.is_dir.return_value = True
        mock_skill_dir1.name = "test_skill"
        mock_skill_file1 = Mock()
        mock_skill_file1.exists.return_value = True
        mock_skill_dir1.__truediv__ = Mock(return_value=mock_skill_file1)

        mock_skill_dir2 = Mock()
        mock_skill_dir2.is_dir.return_value = True
        mock_skill_dir2.name = "__pycache__"  # Should be skipped
        mock_skill_file2 = Mock()
        mock_skill_file2.exists.return_value = True
        mock_skill_dir2.__truediv__ = Mock(return_value=mock_skill_file2)

        mock_file = Mock()
        mock_file.is_dir.return_value = False

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = [mock_skill_dir1, mock_skill_dir2, mock_file]

        with patch('signalwire_agents.skills.registry.Path') as mock_path_cls:
            mock_path_cls.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_skill_on_demand', return_value=None) as mock_load:
                registry.list_skills()

                # Should only load from test_skill directory (not __pycache__ or files)
                mock_load.assert_called_once_with("test_skill")

    def test_load_skill_on_demand_searches_paths(self):
        """Test that _load_skill_on_demand searches built-in and external paths"""
        registry = SkillRegistry()

        with patch.object(registry, '_load_entry_points'):
            with patch.object(registry, '_load_skill_from_path', return_value=None) as mock_load_path:
                result = registry._load_skill_on_demand("nonexistent_skill")

                assert result is None
                # Should have tried loading from the built-in skills directory
                assert mock_load_path.call_count >= 1


class TestSkillLoading:
    """Test skill loading functionality via _load_skill_from_path"""

    def test_load_skill_from_path_no_skill_file(self):
        """Test loading from path where skill.py does not exist"""
        registry = SkillRegistry()

        mock_base_path = Mock()
        mock_skill_dir = Mock()
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = False
        mock_skill_dir.__truediv__ = Mock(return_value=mock_skill_file)
        mock_base_path.__truediv__ = Mock(return_value=mock_skill_dir)

        result = registry._load_skill_from_path("test_skill", mock_base_path)

        assert result is None
        assert len(registry._skills) == 0

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_load_skill_from_path_success(self, mock_getmembers, mock_module_from_spec, mock_spec_from_file):
        """Test successful skill loading from path"""
        registry = SkillRegistry()

        # Mock base_path / skill_name / "skill.py"
        mock_base_path = Mock()
        mock_base_path.name = "skills"
        mock_skill_dir = Mock()
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = True
        mock_skill_dir.__truediv__ = Mock(return_value=mock_skill_file)
        mock_base_path.__truediv__ = Mock(return_value=mock_skill_dir)

        # Mock importlib components
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module

        # Mock inspect.getmembers to return our mock skill
        # _load_skill_from_path checks obj.SKILL_NAME == skill_name
        mock_getmembers.return_value = [
            ("MockSkill", MockSkill),
            ("SomeOtherClass", str),  # Should be ignored
        ]

        with patch.object(registry, 'register_skill') as mock_register:
            result = registry._load_skill_from_path("mock_skill", mock_base_path)

            # Should register the matching skill
            mock_register.assert_called_once_with(MockSkill)

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    def test_load_skill_from_path_import_error(self, mock_spec_from_file):
        """Test skill loading with import error"""
        registry = SkillRegistry()

        # Mock base_path / skill_name / "skill.py"
        mock_base_path = Mock()
        mock_base_path.name = "skills"
        mock_skill_dir = Mock()
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = True
        mock_skill_dir.__truediv__ = Mock(return_value=mock_skill_file)
        mock_base_path.__truediv__ = Mock(return_value=mock_skill_dir)

        # Mock import error
        mock_spec_from_file.side_effect = ImportError("Module not found")

        with patch.object(registry.logger, 'error') as mock_error:
            result = registry._load_skill_from_path("test_skill", mock_base_path)

            assert result is None
            mock_error.assert_called_once()
            assert "Failed to load skill" in mock_error.call_args[0][0]

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    def test_load_skill_from_path_execution_error(self, mock_module_from_spec, mock_spec_from_file):
        """Test skill loading with module execution error"""
        registry = SkillRegistry()

        # Mock base_path / skill_name / "skill.py"
        mock_base_path = Mock()
        mock_base_path.name = "skills"
        mock_skill_dir = Mock()
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = True
        mock_skill_dir.__truediv__ = Mock(return_value=mock_skill_file)
        mock_base_path.__truediv__ = Mock(return_value=mock_skill_dir)

        # Mock importlib components
        mock_spec = Mock()
        mock_loader = Mock()
        mock_loader.exec_module.side_effect = RuntimeError("Execution failed")
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec

        mock_module = Mock()
        mock_module_from_spec.return_value = mock_module

        with patch.object(registry.logger, 'error') as mock_error:
            result = registry._load_skill_from_path("test_skill", mock_base_path)

            assert result is None
            mock_error.assert_called_once()
            assert "Failed to load skill" in mock_error.call_args[0][0]


class TestGlobalRegistry:
    """Test global registry instance"""
    
    def test_global_registry_exists(self):
        """Test that global registry instance exists"""
        assert skill_registry is not None
        assert isinstance(skill_registry, SkillRegistry)
    
    def test_global_registry_singleton_behavior(self):
        """Test that global registry behaves like a singleton"""
        # Import again to get the same instance
        from signalwire_agents.skills.registry import skill_registry as registry2
        
        assert skill_registry is registry2


class TestSkillRegistryIntegration:
    """Test integration scenarios"""
    
    def test_complete_skill_workflow(self):
        """Test complete skill registration and retrieval workflow"""
        registry = SkillRegistry()

        # Register skills
        registry.register_skill(MockSkill)
        registry.register_skill(AnotherMockSkill)

        # Mock filesystem for list_skills and on-demand loading for get_skill_class
        mock_dir1 = Mock()
        mock_dir1.is_dir.return_value = True
        mock_dir1.name = "mock_skill"
        mock_skill_file1 = Mock()
        mock_skill_file1.exists.return_value = True
        mock_dir1.__truediv__ = Mock(return_value=mock_skill_file1)

        mock_dir2 = Mock()
        mock_dir2.is_dir.return_value = True
        mock_dir2.name = "another_mock_skill"
        mock_skill_file2 = Mock()
        mock_skill_file2.exists.return_value = True
        mock_dir2.__truediv__ = Mock(return_value=mock_skill_file2)

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = [mock_dir1, mock_dir2]

        with patch('signalwire_agents.skills.registry.Path') as mock_path_cls:
            mock_path_cls.return_value.parent = mock_skills_dir

            # List all skills
            skills = registry.list_skills()
            assert len(skills) == 2

        # Get specific skills (already registered, no filesystem access needed)
        mock_skill = registry.get_skill_class("mock_skill")
        assert mock_skill == MockSkill

        another_skill = registry.get_skill_class("another_mock_skill")
        assert another_skill == AnotherMockSkill

        # Try to get nonexistent skill
        with patch.object(registry, '_load_skill_on_demand', return_value=None):
            nonexistent = registry.get_skill_class("nonexistent")
            assert nonexistent is None
    
    def test_skill_metadata_completeness(self):
        """Test that skill metadata is complete and correct"""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)

        # Mock filesystem so list_skills finds mock_skill directory
        mock_dir = Mock()
        mock_dir.is_dir.return_value = True
        mock_dir.name = "mock_skill"
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = True
        mock_dir.__truediv__ = Mock(return_value=mock_skill_file)

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = [mock_dir]

        with patch('signalwire_agents.skills.registry.Path') as mock_path_cls:
            mock_path_cls.return_value.parent = mock_skills_dir
            skills = registry.list_skills()
            skill_info = skills[0]

            # Verify all expected fields are present
            expected_fields = [
                "name", "description", "version",
                "required_packages", "required_env_vars",
                "supports_multiple_instances"
            ]

            for field in expected_fields:
                assert field in skill_info

            # Verify field values
            assert skill_info["name"] == "mock_skill"
            assert skill_info["description"] == "A mock skill for testing"
            assert skill_info["version"] == "1.0.0"
            assert skill_info["required_packages"] == ["requests"]
            assert skill_info["required_env_vars"] == ["API_KEY"]
            assert skill_info["supports_multiple_instances"] is True
    
    def test_registry_state_isolation(self):
        """Test that different registry instances are isolated"""
        registry1 = SkillRegistry()
        registry2 = SkillRegistry()
        
        registry1.register_skill(MockSkill)
        
        # registry2 should not have the skill
        assert len(registry1._skills) == 1
        assert len(registry2._skills) == 0
        
        # But both should be able to register skills independently
        registry2.register_skill(AnotherMockSkill)
        
        assert "mock_skill" in registry1._skills
        assert "mock_skill" not in registry2._skills
        assert "another_mock_skill" not in registry1._skills
        assert "another_mock_skill" in registry2._skills
    
    def test_error_recovery(self):
        """Test that registry can recover from errors"""
        registry = SkillRegistry()

        # Register a valid skill
        registry.register_skill(MockSkill)

        # Try to load from a bad path (should not affect existing skills)
        mock_base_path = Mock()
        mock_base_path.name = "skills"
        mock_skill_dir = Mock()
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = True
        mock_skill_dir.__truediv__ = Mock(return_value=mock_skill_file)
        mock_base_path.__truediv__ = Mock(return_value=mock_skill_dir)

        with patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location', side_effect=Exception("Bad import")):
            with patch.object(registry.logger, 'error'):
                result = registry._load_skill_from_path("bad_skill", mock_base_path)

        assert result is None

        # Original skill should still be there
        assert "mock_skill" in registry._skills
        assert registry.get_skill_class("mock_skill") == MockSkill

        # Should still be able to register new skills
        registry.register_skill(AnotherMockSkill)
        assert len(registry._skills) == 2