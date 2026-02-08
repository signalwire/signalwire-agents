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


# ---------------------------------------------------------------------------
# Helper mock skills used by new test classes
# ---------------------------------------------------------------------------

class _SecondMockSkill(SkillBase):
    """A second mock skill with a different name for multi-match testing"""
    SKILL_NAME = "second_mock"
    SKILL_DESCRIPTION = "Second mock"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []
    REQUIRED_ENV_VARS = []
    SUPPORTS_MULTIPLE_INSTANCES = False

    @classmethod
    def get_parameter_schema(cls):
        schema = super().get_parameter_schema()
        schema["extra"] = {"type": "string", "description": "extra", "required": False}
        return schema

    def setup(self):
        pass

    def register_tools(self):
        pass


class _NoParamSchemaSkill(SkillBase):
    """Skill that does not override get_parameter_schema (uses base only)"""
    SKILL_NAME = "no_param_schema"
    SKILL_DESCRIPTION = "No param schema"
    SUPPORTS_MULTIPLE_INSTANCES = False

    def setup(self):
        pass

    def register_tools(self):
        pass


# ---------------------------------------------------------------------------
# TestListAllSkillSources
# ---------------------------------------------------------------------------

class TestListAllSkillSources:
    """Test listing built-in + external skill directories via list_all_skill_sources."""

    def _make_dir_entry(self, name, has_skill_py=True, is_dir=True):
        """Helper to create a mock directory entry."""
        entry = Mock()
        entry.is_dir.return_value = is_dir
        entry.name = name
        skill_file = Mock()
        skill_file.exists.return_value = has_skill_py
        entry.__truediv__ = Mock(return_value=skill_file)
        return entry

    def test_empty_registry_returns_all_categories(self):
        """list_all_skill_sources returns dict with expected keys even when empty."""
        registry = SkillRegistry()
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []
        with patch.object(Path, '__new__', return_value=mock_skills_dir):
            # Easier: patch the parent property used inside the method
            with patch('signalwire_agents.skills.registry.Path') as MockPath:
                # Path(__file__).parent  ->  mock_skills_dir
                MockPath.return_value.parent = mock_skills_dir
                sources = registry.list_all_skill_sources()
        assert set(sources.keys()) == {'built-in', 'external_paths', 'entry_points', 'registered'}
        assert sources['built-in'] == []
        assert sources['external_paths'] == []
        assert sources['registered'] == []

    def test_builtin_skills_listed(self):
        """Built-in skill directories with skill.py are listed."""
        registry = SkillRegistry()
        entries = [
            self._make_dir_entry("weather"),
            self._make_dir_entry("math"),
        ]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert sorted(sources['built-in']) == ['math', 'weather']

    def test_builtin_skips_dunder_dirs(self):
        """Directories starting with __ are excluded from built-in list."""
        registry = SkillRegistry()
        entries = [
            self._make_dir_entry("__pycache__"),
            self._make_dir_entry("__init__"),
            self._make_dir_entry("real_skill"),
        ]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert sources['built-in'] == ['real_skill']

    def test_builtin_skips_non_dir_items(self):
        """Non-directory items in the skills folder are ignored."""
        registry = SkillRegistry()
        entries = [
            self._make_dir_entry("not_a_dir", is_dir=False),
            self._make_dir_entry("a_skill"),
        ]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert sources['built-in'] == ['a_skill']

    def test_builtin_skips_dirs_without_skill_py(self):
        """Directories that lack skill.py are excluded."""
        registry = SkillRegistry()
        entries = [
            self._make_dir_entry("no_skill_file", has_skill_py=False),
            self._make_dir_entry("valid_skill", has_skill_py=True),
        ]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert sources['built-in'] == ['valid_skill']

    def test_external_paths_listed(self):
        """Skills from external directories appear under external_paths."""
        registry = SkillRegistry()

        ext_path = Mock()
        ext_path.exists.return_value = True
        ext_entries = [self._make_dir_entry("custom_skill")]
        ext_path.iterdir.return_value = ext_entries
        registry._external_paths = [ext_path]

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert sources['external_paths'] == ['custom_skill']

    def test_external_path_not_exists_skipped(self):
        """External paths that don't exist are silently skipped."""
        registry = SkillRegistry()

        ext_path = Mock()
        ext_path.exists.return_value = False
        registry._external_paths = [ext_path]

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert sources['external_paths'] == []

    def test_registered_skills_not_in_builtin(self):
        """Registered skills that are NOT in the built-in list go under 'registered'."""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert 'mock_skill' in sources['registered']

    def test_registered_skill_also_builtin_not_duplicated(self):
        """A registered skill whose name matches a built-in should NOT appear in 'registered'."""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)

        # Simulate built-in dir containing 'mock_skill'
        entries = [self._make_dir_entry("mock_skill")]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            sources = registry.list_all_skill_sources()

        assert 'mock_skill' in sources['built-in']
        assert 'mock_skill' not in sources['registered']


# ---------------------------------------------------------------------------
# TestLoadSkillFromPathVariants
# ---------------------------------------------------------------------------

class TestLoadSkillFromPathVariants:
    """Test loading skills from paths - edge cases and variants."""

    def _make_base_path(self, name="skills", skill_file_exists=True):
        """Helper: create mock base_path where base_path/skill_name/skill.py exists."""
        mock_base_path = Mock()
        mock_base_path.name = name
        mock_skill_dir = Mock()
        mock_skill_file = Mock()
        mock_skill_file.exists.return_value = skill_file_exists
        mock_skill_dir.__truediv__ = Mock(return_value=mock_skill_file)
        mock_base_path.__truediv__ = Mock(return_value=mock_skill_dir)
        return mock_base_path

    def test_skill_file_not_exists_returns_none(self):
        """If skill.py does not exist at the expected path, return None."""
        registry = SkillRegistry()
        base = self._make_base_path(skill_file_exists=False)
        assert registry._load_skill_from_path("anything", base) is None

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location', return_value=None)
    def test_spec_is_none_returns_none(self, _mock_spec):
        """When spec_from_file_location returns None, exception is caught and None returned."""
        registry = SkillRegistry()
        base = self._make_base_path()
        result = registry._load_skill_from_path("test_skill", base)
        assert result is None

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_no_matching_skillbase_subclass(self, mock_members, mock_mod, mock_spec):
        """When module has no SkillBase subclass with matching name, logs warning and returns None."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader = Mock()
        mock_spec.return_value = spec
        mock_mod.return_value = Mock()
        # Return a regular class, not a SkillBase subclass
        mock_members.return_value = [("Foo", str), ("Bar", int)]

        with patch.object(registry.logger, 'warning') as warn:
            result = registry._load_skill_from_path("test_skill", base)
            assert result is None
            warn.assert_called_once()
            assert "No skill class found" in warn.call_args[0][0]

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_class_with_wrong_skill_name_skipped(self, mock_members, mock_mod, mock_spec):
        """A SkillBase subclass with a different SKILL_NAME is not loaded."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader = Mock()
        mock_spec.return_value = spec
        mock_mod.return_value = Mock()
        # MockSkill has SKILL_NAME = "mock_skill", but we're searching for "other"
        mock_members.return_value = [("MockSkill", MockSkill)]

        result = registry._load_skill_from_path("other", base)
        assert result is None

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_first_matching_class_wins(self, mock_members, mock_mod, mock_spec):
        """When multiple SkillBase subclasses match, the first one found is returned."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader = Mock()
        mock_spec.return_value = spec
        mock_mod.return_value = Mock()

        # Both have SKILL_NAME = "mock_skill"
        class DuplicateMockSkill(SkillBase):
            SKILL_NAME = "mock_skill"
            SKILL_DESCRIPTION = "Dup"
            SUPPORTS_MULTIPLE_INSTANCES = False
            @classmethod
            def get_parameter_schema(cls):
                schema = super().get_parameter_schema()
                schema["x"] = {"type": "string", "description": "x"}
                return schema
            def setup(self): pass
            def register_tools(self): pass

        mock_members.return_value = [
            ("MockSkill", MockSkill),
            ("DuplicateMockSkill", DuplicateMockSkill),
        ]

        with patch.object(registry, 'register_skill'):
            result = registry._load_skill_from_path("mock_skill", base)
        assert result is MockSkill

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_module_added_to_sys_modules(self, mock_members, mock_mod, mock_spec):
        """The loaded module is inserted into sys.modules."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader = Mock()
        mock_spec.return_value = spec
        fake_module = Mock()
        mock_mod.return_value = fake_module
        mock_members.return_value = [("MockSkill", MockSkill)]

        module_name = f"signalwire_agents_external.skills.mock_skill.skill"
        try:
            with patch.object(registry, 'register_skill'):
                registry._load_skill_from_path("mock_skill", base)
            assert module_name in sys.modules
            assert sys.modules[module_name] is fake_module
        finally:
            sys.modules.pop(module_name, None)

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    def test_exec_module_exception_returns_none(self, mock_mod, mock_spec):
        """If exec_module raises, the error is caught and None is returned."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader.exec_module.side_effect = SyntaxError("bad code")
        mock_spec.return_value = spec
        mock_mod.return_value = Mock()

        result = registry._load_skill_from_path("bad_skill", base)
        assert result is None

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_skillbase_itself_is_skipped(self, mock_members, mock_mod, mock_spec):
        """SkillBase class itself (obj == SkillBase) is skipped during scanning."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader = Mock()
        mock_spec.return_value = spec
        mock_mod.return_value = Mock()
        mock_members.return_value = [("SkillBase", SkillBase)]

        result = registry._load_skill_from_path("SkillBase", base)
        assert result is None

    @patch('signalwire_agents.skills.registry.importlib.util.spec_from_file_location')
    @patch('signalwire_agents.skills.registry.importlib.util.module_from_spec')
    @patch('signalwire_agents.skills.registry.inspect.getmembers')
    def test_class_without_skill_name_attr_skipped(self, mock_members, mock_mod, mock_spec):
        """A class that inherits SkillBase but has no SKILL_NAME attribute is skipped."""
        registry = SkillRegistry()
        base = self._make_base_path()

        spec = Mock()
        spec.loader = Mock()
        mock_spec.return_value = spec
        mock_mod.return_value = Mock()

        # Create a mock class that looks like SkillBase subclass but no SKILL_NAME
        fake_cls = type('FakeSkill', (SkillBase,), {
            'setup': lambda self: None,
            'register_tools': lambda self: None,
        })
        # SkillBase sets SKILL_NAME = None, and the code checks hasattr + obj.SKILL_NAME == skill_name
        mock_members.return_value = [("FakeSkill", fake_cls)]

        result = registry._load_skill_from_path("something", base)
        # SKILL_NAME is None, so obj.SKILL_NAME == "something" is False
        assert result is None


# ---------------------------------------------------------------------------
# TestDirectoryScanning
# ---------------------------------------------------------------------------

class TestDirectoryScanning:
    """Test scanning directories for skills - various scenarios."""

    def _make_dir_entry(self, name, has_skill_py=True, is_dir=True):
        entry = Mock()
        entry.is_dir.return_value = is_dir
        entry.name = name
        skill_file = Mock()
        skill_file.exists.return_value = has_skill_py
        entry.__truediv__ = Mock(return_value=skill_file)
        return entry

    # -- add_skill_directory tests --

    def test_add_skill_directory_valid(self, tmp_path):
        """A valid directory is added to external paths."""
        registry = SkillRegistry()
        registry.add_skill_directory(str(tmp_path))
        assert Path(tmp_path) in registry._external_paths

    def test_add_skill_directory_not_exists(self):
        """Non-existent path raises ValueError."""
        registry = SkillRegistry()
        with pytest.raises(ValueError, match="does not exist"):
            registry.add_skill_directory("/no/such/path/abc123")

    def test_add_skill_directory_not_a_dir(self, tmp_path):
        """A file (not directory) raises ValueError."""
        f = tmp_path / "afile.txt"
        f.write_text("hello")
        registry = SkillRegistry()
        with pytest.raises(ValueError, match="not a directory"):
            registry.add_skill_directory(str(f))

    def test_add_skill_directory_duplicate_ignored(self, tmp_path):
        """Adding the same directory twice only stores it once."""
        registry = SkillRegistry()
        registry.add_skill_directory(str(tmp_path))
        registry.add_skill_directory(str(tmp_path))
        assert registry._external_paths.count(Path(tmp_path)) == 1

    # -- get_all_skills_schema tests --

    def test_get_all_skills_schema_includes_registered(self):
        """Already registered skills appear in schema under 'registered'."""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                schema = registry.get_all_skills_schema()

        assert 'mock_skill' in schema
        assert schema['mock_skill']['source'] == 'registered'
        assert schema['mock_skill']['name'] == 'mock_skill'
        assert schema['mock_skill']['description'] == 'A mock skill for testing'

    def test_get_all_skills_schema_builtin_scan(self):
        """Built-in skills are scanned and added with source='built-in'."""
        registry = SkillRegistry()

        entries = [self._make_dir_entry("weather")]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                with patch.object(registry, '_load_skill_on_demand', return_value=MockSkill):
                    schema = registry.get_all_skills_schema()

        assert 'mock_skill' in schema
        assert schema['mock_skill']['source'] == 'built-in'

    def test_get_all_skills_schema_external_scan(self, tmp_path):
        """External path skills appear with source='external'."""
        registry = SkillRegistry()

        ext_entry = self._make_dir_entry("ext_skill")
        ext_path = Mock()
        ext_path.exists.return_value = True
        ext_path.iterdir.return_value = [ext_entry]
        registry._external_paths = [ext_path]

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                with patch.object(registry, '_load_skill_on_demand', return_value=MockSkill):
                    schema = registry.get_all_skills_schema()

        assert 'mock_skill' in schema
        assert schema['mock_skill']['source'] == 'external'

    def test_get_all_skills_schema_env_paths(self):
        """Skills from SIGNALWIRE_SKILL_PATHS env var are scanned."""
        registry = SkillRegistry()

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        env_dir_entry = self._make_dir_entry("env_skill")
        mock_env_path = Mock()
        mock_env_path.exists.return_value = True
        mock_env_path.iterdir.return_value = [env_dir_entry]

        # Path(__file__) should return an object whose .parent yields mock_skills_dir
        # Path(path_str) for env paths should return mock_env_path
        mock_file_path = Mock()
        mock_file_path.parent = mock_skills_dir

        def path_factory(x):
            if x == '/fake/env/path':
                return mock_env_path
            # For __file__ and anything else, return something with .parent
            result = Mock()
            result.parent = mock_skills_dir
            return result

        with patch('signalwire_agents.skills.registry.Path', side_effect=path_factory):
            with patch.object(registry, '_load_entry_points'):
                with patch.object(registry, '_load_skill_on_demand', return_value=MockSkill):
                    with patch.dict('os.environ', {'SIGNALWIRE_SKILL_PATHS': '/fake/env/path'}):
                        schema = registry.get_all_skills_schema()

        assert 'mock_skill' in schema

    def test_get_all_skills_schema_skips_already_in_schema(self):
        """Skills already present in schema from an earlier source are not overwritten."""
        registry = SkillRegistry()
        registry.register_skill(MockSkill)

        # Built-in also has 'mock_skill' dir
        entries = [self._make_dir_entry("mock_skill")]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                schema = registry.get_all_skills_schema()

        # Should be 'registered', not overwritten to 'built-in'
        assert schema['mock_skill']['source'] == 'registered'

    def test_get_all_skills_schema_handles_load_failure(self):
        """When _load_skill_on_demand raises, error is logged and skill skipped."""
        registry = SkillRegistry()

        entries = [self._make_dir_entry("bad_skill")]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                with patch.object(registry, '_load_skill_on_demand', side_effect=RuntimeError("boom")):
                    with patch.object(registry.logger, 'error'):
                        schema = registry.get_all_skills_schema()

        assert 'bad_skill' not in schema

    def test_get_all_skills_schema_handles_none_from_load(self):
        """When _load_skill_on_demand returns None, the skill is simply skipped."""
        registry = SkillRegistry()

        entries = [self._make_dir_entry("missing_skill")]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                with patch.object(registry, '_load_skill_on_demand', return_value=None):
                    schema = registry.get_all_skills_schema()

        assert 'missing_skill' not in schema

    def test_get_all_skills_schema_skill_without_get_parameter_schema(self):
        """If skill_class lacks get_parameter_schema, empty dict is used for parameters."""
        registry = SkillRegistry()

        # Create a skill class that raises AttributeError on get_parameter_schema
        fake_skill = Mock()
        fake_skill.SKILL_NAME = "attr_err_skill"
        fake_skill.SKILL_DESCRIPTION = "desc"
        fake_skill.SKILL_VERSION = "1.0.0"
        fake_skill.SUPPORTS_MULTIPLE_INSTANCES = False
        fake_skill.REQUIRED_PACKAGES = []
        fake_skill.REQUIRED_ENV_VARS = []
        fake_skill.get_parameter_schema.side_effect = AttributeError("nope")

        entries = [self._make_dir_entry("attr_err_skill")]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                with patch.object(registry, '_load_skill_on_demand', return_value=fake_skill):
                    schema = registry.get_all_skills_schema()

        assert 'attr_err_skill' in schema
        assert schema['attr_err_skill']['parameters'] == {}

    def test_get_all_skills_schema_external_path_not_exists(self):
        """External paths that don't exist are silently skipped in schema scan."""
        registry = SkillRegistry()

        ext_path = Mock()
        ext_path.exists.return_value = False
        registry._external_paths = [ext_path]

        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = []

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            with patch.object(registry, '_load_entry_points'):
                schema = registry.get_all_skills_schema()

        assert schema == {}

    def test_list_skills_skips_dirs_without_skill_py(self):
        """list_skills ignores directories that do not contain skill.py."""
        registry = SkillRegistry()

        entries = [self._make_dir_entry("no_py", has_skill_py=False)]
        mock_skills_dir = Mock()
        mock_skills_dir.iterdir.return_value = entries

        with patch('signalwire_agents.skills.registry.Path') as MockPath:
            MockPath.return_value.parent = mock_skills_dir
            skills = registry.list_skills()

        assert skills == []

    def test_deprecated_load_skill_from_directory_noop(self):
        """_load_skill_from_directory is a no-op for backwards compat."""
        registry = SkillRegistry()
        registry._load_skill_from_directory(Path("/some/path"))
        assert registry._skills == {}


# ---------------------------------------------------------------------------
# TestEntryPointLoading
# ---------------------------------------------------------------------------

class TestEntryPointLoading:
    """Test loading skills from entry points."""

    def test_valid_entry_point_with_select(self):
        """Entry points loaded via .select() for Python 3.12+ API."""
        registry = SkillRegistry()

        mock_ep = Mock()
        mock_ep.name = "mock_skill"
        mock_ep.load.return_value = MockSkill

        mock_all_eps = Mock()
        mock_all_eps.select.return_value = [mock_ep]
        # hasattr(mock_all_eps, 'select') -> True by default with Mock

        with patch('importlib.metadata.entry_points', return_value=mock_all_eps):
            with patch.object(registry, 'register_skill') as mock_reg:
                registry._load_entry_points()
                mock_reg.assert_called_once_with(MockSkill)

        assert registry._entry_points_loaded is True

    def test_valid_entry_point_dict_api(self):
        """Entry points loaded via dict .get() for older Python API."""
        registry = SkillRegistry()

        mock_ep = Mock()
        mock_ep.name = "mock_skill"
        mock_ep.load.return_value = MockSkill

        # Create object without 'select' attribute
        mock_all_eps = {'signalwire_agents.skills': [mock_ep]}

        with patch('importlib.metadata.entry_points', return_value=mock_all_eps):
            with patch.object(registry, 'register_skill') as mock_reg:
                registry._load_entry_points()
                mock_reg.assert_called_once_with(MockSkill)

    def test_broken_entry_point_load_fails(self):
        """If entry_point.load() raises, error is logged and loading continues."""
        registry = SkillRegistry()

        mock_ep = Mock()
        mock_ep.name = "broken_ep"
        mock_ep.load.side_effect = ImportError("can't load")

        mock_all_eps = Mock()
        mock_all_eps.select.return_value = [mock_ep]

        with patch('importlib.metadata.entry_points', return_value=mock_all_eps):
            with patch.object(registry.logger, 'error') as mock_err:
                registry._load_entry_points()
                mock_err.assert_called_once()
                assert "Failed to load skill from entry point" in mock_err.call_args[0][0]

    def test_non_skillbase_entry_point(self):
        """Entry points that load a non-SkillBase class are warned about."""
        registry = SkillRegistry()

        mock_ep = Mock()
        mock_ep.name = "not_a_skill"
        mock_ep.load.return_value = str  # Not a SkillBase subclass

        mock_all_eps = Mock()
        mock_all_eps.select.return_value = [mock_ep]

        with patch('importlib.metadata.entry_points', return_value=mock_all_eps):
            with patch.object(registry.logger, 'warning') as mock_warn:
                registry._load_entry_points()
                mock_warn.assert_called_once()
                assert "does not provide a SkillBase subclass" in mock_warn.call_args[0][0]

    def test_entry_points_loaded_flag_prevents_reload(self):
        """Once _entry_points_loaded is True, the method returns immediately."""
        registry = SkillRegistry()
        registry._entry_points_loaded = True

        with patch('importlib.metadata.entry_points') as mock_ep:
            registry._load_entry_points()
            mock_ep.assert_not_called()

    def test_entry_points_overall_exception(self):
        """If entry_points() itself raises, the error is caught gracefully."""
        registry = SkillRegistry()

        with patch('importlib.metadata.entry_points', side_effect=Exception("metadata broke")):
            with patch.object(registry.logger, 'debug') as mock_debug:
                registry._load_entry_points()
                mock_debug.assert_called_once()
                assert "Entry point loading failed" in mock_debug.call_args[0][0]

        # Flag should still be set to prevent retries
        assert registry._entry_points_loaded is True

    def test_multiple_entry_points_loaded(self):
        """Multiple valid entry points are all loaded."""
        registry = SkillRegistry()

        ep1 = Mock()
        ep1.name = "mock_skill"
        ep1.load.return_value = MockSkill

        ep2 = Mock()
        ep2.name = "another"
        ep2.load.return_value = AnotherMockSkill

        mock_all_eps = Mock()
        mock_all_eps.select.return_value = [ep1, ep2]

        with patch('importlib.metadata.entry_points', return_value=mock_all_eps):
            with patch.object(registry, 'register_skill') as mock_reg:
                registry._load_entry_points()
                assert mock_reg.call_count == 2

    def test_entry_point_register_skill_failure(self):
        """If register_skill raises for an entry point, error is logged."""
        registry = SkillRegistry()

        mock_ep = Mock()
        mock_ep.name = "fail_register"
        mock_ep.load.return_value = MockSkill

        mock_all_eps = Mock()
        mock_all_eps.select.return_value = [mock_ep]

        with patch('importlib.metadata.entry_points', return_value=mock_all_eps):
            with patch.object(registry, 'register_skill', side_effect=ValueError("bad skill")):
                with patch.object(registry.logger, 'error') as mock_err:
                    registry._load_entry_points()
                    mock_err.assert_called_once()

    # -- _load_skill_on_demand integration with entry points --

    def test_load_on_demand_returns_cached_skill(self):
        """If skill is already in _skills, it is returned immediately without loading."""
        registry = SkillRegistry()
        registry._skills["mock_skill"] = MockSkill

        result = registry._load_skill_on_demand("mock_skill")
        assert result is MockSkill

    def test_load_on_demand_finds_skill_after_entry_points(self):
        """If entry points load the skill, it is returned without path scanning."""
        registry = SkillRegistry()

        def fake_load_eps():
            registry._entry_points_loaded = True
            registry._skills["dynamic"] = MockSkill

        with patch.object(registry, '_load_entry_points', side_effect=fake_load_eps):
            result = registry._load_skill_on_demand("dynamic")

        assert result is MockSkill

    def test_load_on_demand_searches_external_paths(self):
        """_load_skill_on_demand searches registered external paths."""
        registry = SkillRegistry()
        ext_path = Path("/fake/external")
        registry._external_paths = [ext_path]

        call_log = []

        def fake_load_from_path(name, path):
            call_log.append((name, path))
            if path == ext_path:
                return MockSkill
            return None

        with patch.object(registry, '_load_entry_points'):
            with patch.object(registry, '_load_skill_from_path', side_effect=fake_load_from_path):
                result = registry._load_skill_on_demand("mock_skill")

        assert result is MockSkill
        # Should have tried built-in first, then external
        paths_tried = [c[1] for c in call_log]
        assert ext_path in paths_tried

    def test_load_on_demand_searches_env_paths(self):
        """_load_skill_on_demand searches SIGNALWIRE_SKILL_PATHS env var."""
        registry = SkillRegistry()

        call_log = []

        def fake_load_from_path(name, path):
            call_log.append((name, path))
            if str(path) == "/env/skills":
                return MockSkill
            return None

        with patch.object(registry, '_load_entry_points'):
            with patch.object(registry, '_load_skill_from_path', side_effect=fake_load_from_path):
                with patch.dict('os.environ', {'SIGNALWIRE_SKILL_PATHS': '/env/skills'}):
                    result = registry._load_skill_on_demand("mock_skill")

        assert result is MockSkill

    def test_load_on_demand_env_path_empty_string_skipped(self):
        """Empty strings in SIGNALWIRE_SKILL_PATHS are skipped."""
        registry = SkillRegistry()

        with patch.object(registry, '_load_entry_points'):
            with patch.object(registry, '_load_skill_from_path', return_value=None) as mock_load:
                with patch.dict('os.environ', {'SIGNALWIRE_SKILL_PATHS': ''}):
                    result = registry._load_skill_on_demand("skill_x")

        assert result is None
        # _load_skill_from_path should only be called for the built-in dir, not for empty string
        for call in mock_load.call_args_list:
            assert call[0][0] == "skill_x"  # First arg is skill name
            # Second arg should not be Path('')
            assert str(call[0][1]) != ''

    def test_load_on_demand_not_found_returns_none(self):
        """When skill is not found anywhere, None is returned with debug log."""
        registry = SkillRegistry()

        with patch.object(registry, '_load_entry_points'):
            with patch.object(registry, '_load_skill_from_path', return_value=None):
                with patch.dict('os.environ', {'SIGNALWIRE_SKILL_PATHS': ''}):
                    with patch.object(registry.logger, 'debug') as mock_debug:
                        result = registry._load_skill_on_demand("nowhere_skill")

        assert result is None
        mock_debug.assert_called_once()
        assert "not found" in mock_debug.call_args[0][0]


# ---------------------------------------------------------------------------
# TestRegisterSkillValidation
# ---------------------------------------------------------------------------

class TestRegisterSkillValidation:
    """Test register_skill validation edge cases."""

    def test_register_non_subclass_raises(self):
        """Registering a class that doesn't inherit SkillBase raises ValueError."""
        registry = SkillRegistry()
        with pytest.raises(ValueError, match="must inherit from SkillBase"):
            registry.register_skill(str)

    def test_register_no_skill_name_raises(self):
        """Registering a skill with SKILL_NAME=None raises ValueError."""
        registry = SkillRegistry()
        with pytest.raises(ValueError, match="must define SKILL_NAME"):
            registry.register_skill(InvalidSkill)

    def test_register_skill_no_get_parameter_schema_raises(self):
        """Skill without get_parameter_schema method raises ValueError."""
        registry = SkillRegistry()

        class BadSkill(SkillBase):
            SKILL_NAME = "bad"
            SKILL_DESCRIPTION = "bad"
            def setup(self): pass
            def register_tools(self): pass

        # Delete get_parameter_schema to simulate missing method
        # Since SkillBase has it, we need to make it not callable
        with patch.object(BadSkill, 'get_parameter_schema', new_callable=lambda: property(lambda self: None)):
            # The hasattr check or callable check will fail
            with pytest.raises(ValueError):
                registry.register_skill(BadSkill)

    def test_register_skill_schema_returns_non_dict(self):
        """Skill whose get_parameter_schema returns non-dict raises ValueError."""
        registry = SkillRegistry()

        class BadSchemaSkill(SkillBase):
            SKILL_NAME = "bad_schema"
            SKILL_DESCRIPTION = "bad schema"
            def setup(self): pass
            def register_tools(self): pass
            @classmethod
            def get_parameter_schema(cls):
                return "not a dict"

        with pytest.raises(ValueError, match="must return a dictionary"):
            registry.register_skill(BadSchemaSkill)

    def test_register_skill_schema_returns_empty_dict(self):
        """Skill whose get_parameter_schema returns empty dict raises ValueError."""
        registry = SkillRegistry()

        class EmptySchemaSkill(SkillBase):
            SKILL_NAME = "empty_schema"
            SKILL_DESCRIPTION = "empty schema"
            def setup(self): pass
            def register_tools(self): pass
            @classmethod
            def get_parameter_schema(cls):
                return {}

        with pytest.raises(ValueError, match="returned an empty dictionary"):
            registry.register_skill(EmptySchemaSkill)

    def test_register_skill_schema_exception(self):
        """Skill whose get_parameter_schema raises non-ValueError is wrapped."""
        registry = SkillRegistry()

        class ExcSchemaSkill(SkillBase):
            SKILL_NAME = "exc_schema"
            SKILL_DESCRIPTION = "exc schema"
            def setup(self): pass
            def register_tools(self): pass
            @classmethod
            def get_parameter_schema(cls):
                raise RuntimeError("boom")

        with pytest.raises(ValueError, match="failed"):
            registry.register_skill(ExcSchemaSkill)