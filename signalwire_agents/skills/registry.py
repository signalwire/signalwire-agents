"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import importlib
import importlib.util
import inspect
import sys
import threading
from typing import Dict, List, Type, Optional, Any
from pathlib import Path

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.logging_config import get_logger

class SkillRegistry:
    """Global registry for on-demand skill loading"""
    
    def __init__(self):
        self._skills: Dict[str, Type[SkillBase]] = {}
        self._external_paths: List[Path] = []  # Additional paths to search for skills
        self._entry_points_loaded = False
        self._lock = threading.RLock()
        self.logger = get_logger("skill_registry")
    
    def _load_skill_on_demand(self, skill_name: str) -> Optional[Type[SkillBase]]:
        """Load a skill on-demand by name"""
        with self._lock:
            if skill_name in self._skills:
                return self._skills[skill_name]

            # Search in built-in skills directory FIRST (before entry points)
            skills_dir = Path(__file__).parent
            skill_class = self._load_skill_from_path(skill_name, skills_dir)
            if skill_class:
                return skill_class

            # Then load entry points (cannot override built-in skills)
            self._load_entry_points()

            # Check if skill was loaded from entry points
            if skill_name in self._skills:
                return self._skills[skill_name]

            # Search in external paths
            for external_path in self._external_paths:
                skill_class = self._load_skill_from_path(skill_name, external_path)
                if skill_class:
                    return skill_class

            # Search in environment variable paths
            env_paths = os.environ.get('SIGNALWIRE_SKILL_PATHS', '').split(os.pathsep)
            for path_str in env_paths:
                if path_str:
                    skill_class = self._load_skill_from_path(skill_name, Path(path_str))
                    if skill_class:
                        return skill_class

            self.logger.debug(f"Skill '{skill_name}' not found in any registered paths")
            return None
    
    def _load_skill_from_path(self, skill_name: str, base_path: Path) -> Optional[Type[SkillBase]]:
        """Try to load a skill from a specific base path"""
        # Prevent path traversal in skill names
        if '..' in skill_name or os.sep in skill_name or '/' in skill_name:
            self.logger.error(f"Invalid skill name (path traversal attempt): {skill_name}")
            return None

        skill_dir = base_path / skill_name

        # Verify the resolved path stays within the expected base path
        try:
            resolved_skill_dir = skill_dir.resolve()
            resolved_base = base_path.resolve()
            try:
                resolved_skill_dir.relative_to(resolved_base)
            except ValueError:
                self.logger.error(f"Skill path escapes base directory: {skill_name}")
                return None
        except Exception:
            self.logger.error(f"Failed to resolve skill path for: {skill_name}")
            return None
        skill_file = skill_dir / "skill.py"
        
        if not skill_file.exists():
            return None
            
        try:
            # Create unique module name to avoid conflicts
            module_name = f"signalwire_agents_external.{base_path.name}.{skill_name}.skill"
            spec = importlib.util.spec_from_file_location(module_name, skill_file)
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to handle relative imports
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find SkillBase subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, SkillBase) and 
                    obj != SkillBase and
                    hasattr(obj, 'SKILL_NAME') and
                    obj.SKILL_NAME == skill_name):  # Match exact skill name
                    
                    self.register_skill(obj)
                    return obj
                    
            self.logger.warning(f"No skill class found with name '{skill_name}' in {skill_file}")
            return None
                    
        except Exception as e:
            self.logger.error(f"Failed to load skill '{skill_name}' from {skill_file}: {e}")
            return None
    
    def discover_skills(self) -> None:
        """Deprecated: Skills are now loaded on-demand"""
        # Keep this method for backwards compatibility but make it a no-op
        pass
    
    def _load_skill_from_directory(self, skill_dir: Path) -> None:
        """Deprecated: Skills are now loaded on-demand"""
        # Keep this method for backwards compatibility but make it a no-op
        pass
    
    def register_skill(self, skill_class: Type[SkillBase]) -> None:
        """
        Register a skill class directly

        This allows third-party code to register skill classes without
        requiring them to be in a specific directory structure.

        Args:
            skill_class: A class that inherits from SkillBase

        Example:
            from my_custom_skills import MyWeatherSkill
            skill_registry.register_skill(MyWeatherSkill)
        """
        with self._lock:
            if not issubclass(skill_class, SkillBase):
                raise ValueError(f"{skill_class} must inherit from SkillBase")

            if not hasattr(skill_class, 'SKILL_NAME') or skill_class.SKILL_NAME is None:
                raise ValueError(f"{skill_class} must define SKILL_NAME")

            # Validate that the skill has a proper parameter schema
            if not hasattr(skill_class, 'get_parameter_schema') or not callable(getattr(skill_class, 'get_parameter_schema')):
                raise ValueError(f"{skill_class.__name__} must have get_parameter_schema() classmethod")

            # Try to call get_parameter_schema to ensure it's properly implemented
            try:
                schema = skill_class.get_parameter_schema()
                if not isinstance(schema, dict):
                    raise ValueError(f"{skill_class.__name__}.get_parameter_schema() must return a dictionary, got {type(schema)}")

                # Ensure it's not an empty schema (skills should at least have the base parameters)
                if not schema:
                    raise ValueError(f"{skill_class.__name__}.get_parameter_schema() returned an empty dictionary. Skills should at least call super().get_parameter_schema()")

                # Check if the skill has overridden the method (not just inherited base)
                skill_method = getattr(skill_class, 'get_parameter_schema', None)
                base_method = getattr(SkillBase, 'get_parameter_schema', None)

                if skill_method and base_method:
                    # For class methods, check the underlying function
                    skill_func = skill_method.__func__ if hasattr(skill_method, '__func__') else skill_method
                    base_func = base_method.__func__ if hasattr(base_method, '__func__') else base_method

                    if skill_func is base_func:
                        # Get base schema to check if skill added any parameters
                        base_schema = SkillBase.get_parameter_schema()
                        if set(schema.keys()) == set(base_schema.keys()):
                            raise ValueError(f"{skill_class.__name__} must override get_parameter_schema() to define its specific parameters")

            except AttributeError as e:
                raise ValueError(f"{skill_class.__name__} must properly implement get_parameter_schema() classmethod")
            except ValueError:
                raise  # Re-raise our validation errors
            except Exception as e:
                raise ValueError(f"{skill_class.__name__}.get_parameter_schema() failed: {e}")

            if skill_class.SKILL_NAME in self._skills:
                self.logger.warning(f"Skill '{skill_class.SKILL_NAME}' already registered")
                return

            self._skills[skill_class.SKILL_NAME] = skill_class
            self.logger.debug("skill_registered", extra={"skill": skill_class.SKILL_NAME})
    
    def get_skill_class(self, skill_name: str) -> Optional[Type[SkillBase]]:
        """Get skill class by name, loading on-demand if needed"""
        # First check if already loaded
        if skill_name in self._skills:
            return self._skills[skill_name]
        
        # Try to load on-demand
        return self._load_skill_on_demand(skill_name)
    
    def list_skills(self) -> List[Dict[str, str]]:
        """List all available skills by scanning directories (only when explicitly requested)"""
        # Only scan when this method is explicitly called (e.g., for CLI tools)
        skills_dir = Path(__file__).parent
        available_skills = []
        
        for item in skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                skill_file = item / "skill.py"
                if skill_file.exists():
                    # Try to load the skill to get its metadata
                    skill_class = self._load_skill_on_demand(item.name)
                    if skill_class:
                        available_skills.append({
                            "name": skill_class.SKILL_NAME,
                            "description": skill_class.SKILL_DESCRIPTION,
                            "version": skill_class.SKILL_VERSION,
                            "required_packages": skill_class.REQUIRED_PACKAGES,
                            "required_env_vars": skill_class.REQUIRED_ENV_VARS,
                            "supports_multiple_instances": skill_class.SUPPORTS_MULTIPLE_INSTANCES
                        })
        
        return available_skills
    
    def get_all_skills_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete schema for all available skills including parameter metadata
        
        This method scans all available skills and returns a comprehensive schema
        that includes skill metadata and parameter definitions suitable for GUI
        configuration or API documentation.
        
        Returns:
            Dict[str, Dict[str, Any]]: Complete skill schema where keys are skill names
            and values contain:
                - name: Skill name
                - description: Skill description
                - version: Skill version
                - supports_multiple_instances: Whether multiple instances are allowed
                - required_packages: List of required Python packages
                - required_env_vars: List of required environment variables
                - parameters: Parameter schema from get_parameter_schema()
                - source: Where the skill was loaded from ('built-in', 'external', 'entry_point', 'registered')
        
        Example:
            {
                "web_search": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "version": "1.0.0",
                    "supports_multiple_instances": True,
                    "required_packages": ["bs4", "requests"],
                    "required_env_vars": [],
                    "parameters": {
                        "api_key": {
                            "type": "string",
                            "description": "Google API key",
                            "required": True,
                            "hidden": True,
                            "env_var": "GOOGLE_SEARCH_API_KEY"
                        },
                        ...
                    },
                    "source": "built-in"
                }
            }
        """
        skills_schema = {}
        
        # Load entry points first
        self._load_entry_points()
        
        # Helper function to add skill to schema
        def add_skill_to_schema(skill_class, source):
            try:
                # Get parameter schema
                try:
                    parameter_schema = skill_class.get_parameter_schema()
                except AttributeError:
                    # Skill doesn't implement get_parameter_schema yet
                    parameter_schema = {}
                
                skills_schema[skill_class.SKILL_NAME] = {
                    "name": skill_class.SKILL_NAME,
                    "description": skill_class.SKILL_DESCRIPTION,
                    "version": getattr(skill_class, 'SKILL_VERSION', '1.0.0'),
                    "supports_multiple_instances": getattr(skill_class, 'SUPPORTS_MULTIPLE_INSTANCES', False),
                    "required_packages": getattr(skill_class, 'REQUIRED_PACKAGES', []),
                    "required_env_vars": getattr(skill_class, 'REQUIRED_ENV_VARS', []),
                    "parameters": parameter_schema,
                    "source": source
                }
            except Exception as e:
                self.logger.error(f"Failed to get schema for skill '{skill_class.SKILL_NAME}': {e}")
        
        # Add already registered skills first (includes entry points)
        for skill_name, skill_class in self._skills.items():
            add_skill_to_schema(skill_class, 'registered')
        
        # Scan built-in skills directory
        skills_dir = Path(__file__).parent
        for item in skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                skill_file = item / "skill.py"
                if skill_file.exists() and item.name not in skills_schema:
                    try:
                        skill_class = self._load_skill_on_demand(item.name)
                        if skill_class:
                            add_skill_to_schema(skill_class, 'built-in')
                    except Exception as e:
                        self.logger.error(f"Failed to load skill '{item.name}': {e}")
        
        # Scan external directories
        for external_path in self._external_paths:
            if external_path.exists():
                for item in external_path.iterdir():
                    if item.is_dir() and not item.name.startswith('__'):
                        skill_file = item / "skill.py"
                        if skill_file.exists() and item.name not in skills_schema:
                            try:
                                skill_class = self._load_skill_on_demand(item.name)
                                if skill_class:
                                    add_skill_to_schema(skill_class, 'external')
                            except Exception as e:
                                self.logger.error(f"Failed to load skill '{item.name}': {e}")
        
        # Scan environment variable paths
        env_paths = os.environ.get('SIGNALWIRE_SKILL_PATHS', '').split(os.pathsep)
        for path_str in env_paths:
            if path_str:
                env_path = Path(path_str)
                if env_path.exists():
                    for item in env_path.iterdir():
                        if item.is_dir() and not item.name.startswith('__'):
                            skill_file = item / "skill.py"
                            if skill_file.exists() and item.name not in skills_schema:
                                try:
                                    skill_class = self._load_skill_on_demand(item.name)
                                    if skill_class:
                                        add_skill_to_schema(skill_class, 'external')
                                except Exception as e:
                                    self.logger.error(f"Failed to load skill '{item.name}': {e}")
        
        return skills_schema
    
    def add_skill_directory(self, path: str) -> None:
        """
        Add a directory to search for skills

        This allows third-party skill collections to be registered by path.
        Skills in these directories should follow the same structure as built-in skills:
        - Each skill in its own subdirectory
        - skill.py file containing the skill class

        Args:
            path: Path to directory containing skill subdirectories

        Example:
            skill_registry.add_skill_directory('/opt/custom_skills')
            # Now agent.add_skill('my_custom_skill') will search in this directory
        """
        with self._lock:
            skill_path = Path(path)
            if not skill_path.exists():
                raise ValueError(f"Skill directory does not exist: {path}")
            if not skill_path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")

            if skill_path not in self._external_paths:
                self._external_paths.append(skill_path)
                self.logger.info(f"Added external skill directory: {path}")
    
    def _load_entry_points(self) -> None:
        """
        Load skills from Python entry points
        
        This allows installed packages to register skills via setup.py:
        
        entry_points={
            'signalwire_agents.skills': [
                'weather = my_package.skills:WeatherSkill',
                'stock = my_package.skills:StockSkill',
            ]
        }
        """
        if self._entry_points_loaded:
            return
            
        self._entry_points_loaded = True
        
        try:
            from importlib.metadata import entry_points

            all_eps = entry_points()
            if hasattr(all_eps, 'select'):
                # Python 3.12+ returns SelectableGroups; 3.10-3.11 also support select
                skill_entries = all_eps.select(group='signalwire_agents.skills')
            else:
                skill_entries = all_eps.get('signalwire_agents.skills', [])

            # Determine built-in skill names for conflict detection
            builtin_skills_dir = Path(__file__).parent
            builtin_skill_names = set()
            for item in builtin_skills_dir.iterdir():
                if item.is_dir() and not item.name.startswith('__'):
                    skill_file = item / "skill.py"
                    if skill_file.exists():
                        builtin_skill_names.add(item.name)

            for entry_point in skill_entries:
                try:
                    skill_class = entry_point.load()
                    if issubclass(skill_class, SkillBase):
                        # Block entry points from overriding built-in skills
                        skill_name = getattr(skill_class, 'SKILL_NAME', None)
                        if skill_name and (skill_name in builtin_skill_names or skill_name in self._skills):
                            self.logger.error(
                                f"Entry point '{entry_point.name}' tried to register skill '{skill_name}' "
                                f"which conflicts with a built-in skill. Skipping."
                            )
                            continue
                        self.register_skill(skill_class)
                        self.logger.info(f"Loaded skill '{skill_class.SKILL_NAME}' from entry point '{entry_point.name}'")
                    else:
                        self.logger.warning(f"Entry point '{entry_point.name}' does not provide a SkillBase subclass")
                except Exception as e:
                    self.logger.error(f"Failed to load skill from entry point '{entry_point.name}': {e}")

        except Exception as e:
            self.logger.debug(f"Entry point loading failed: {e}")
    
    def list_all_skill_sources(self) -> Dict[str, List[str]]:
        """
        List all skill sources and the skills available from each
        
        Returns a dictionary mapping source types to lists of skill names:
        {
            'built-in': ['datetime', 'math', ...],
            'external_paths': ['custom_skill1', ...],
            'entry_points': ['weather', ...],
            'registered': ['my_skill', ...]
        }
        """
        sources = {
            'built-in': [],
            'external_paths': [],
            'entry_points': [],
            'registered': []
        }
        
        # Built-in skills
        skills_dir = Path(__file__).parent
        for item in skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                skill_file = item / "skill.py"
                if skill_file.exists():
                    sources['built-in'].append(item.name)
        
        # External path skills
        for external_path in self._external_paths:
            if external_path.exists():
                for item in external_path.iterdir():
                    if item.is_dir() and not item.name.startswith('__'):
                        skill_file = item / "skill.py"
                        if skill_file.exists():
                            sources['external_paths'].append(item.name)
        
        # Already registered skills
        for skill_name in self._skills:
            # Determine source of registered skill
            if skill_name not in sources['built-in']:
                sources['registered'].append(skill_name)
        
        return sources

# Global registry instance
skill_registry = SkillRegistry() 