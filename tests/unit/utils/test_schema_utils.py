"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for schema_utils module
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, List, Any, Optional

from signalwire_agents.utils.schema_utils import SchemaUtils


class TestSchemaUtils:
    """Test SchemaUtils functionality"""
    
    def test_basic_initialization_with_schema_path(self):
        """Test basic SchemaUtils initialization with schema path"""
        with patch.object(SchemaUtils, 'load_schema', return_value={}):
            with patch.object(SchemaUtils, '_extract_verb_definitions', return_value={}):
                utils = SchemaUtils(schema_path="/path/to/schema.json")
                
                assert utils.schema_path == "/path/to/schema.json"
                assert utils.schema == {}
                assert utils.verbs == {}
    
    def test_initialization_without_schema_path(self):
        """Test initialization without schema path uses default"""
        with patch.object(SchemaUtils, '_get_default_schema_path', return_value="/default/schema.json"):
            with patch.object(SchemaUtils, 'load_schema', return_value={}):
                with patch.object(SchemaUtils, '_extract_verb_definitions', return_value={}):
                    utils = SchemaUtils()
                    
                    assert utils.schema_path == "/default/schema.json"
    
    def test_get_default_schema_path_importlib_resources_new(self):
        """Test default schema path using importlib.resources (Python 3.13+)"""
        utils = SchemaUtils.__new__(SchemaUtils)  # Create without calling __init__
        
        mock_path = Mock()
        mock_path.__str__ = Mock(return_value="/package/schema.json")
        
        with patch('importlib.resources.files') as mock_files:
            mock_files.return_value.joinpath.return_value = mock_path
            
            result = utils._get_default_schema_path()
            
            assert result == "/package/schema.json"
            mock_files.assert_called_once_with("signalwire_agents")
    
    def test_get_default_schema_path_importlib_resources_old(self):
        """Test default schema path using importlib.resources (Python 3.7-3.8)"""
        utils = SchemaUtils.__new__(SchemaUtils)
        
        with patch('importlib.resources.files', side_effect=AttributeError):
            with patch('importlib.resources.path') as mock_path:
                mock_context = Mock()
                mock_context.__enter__ = Mock(return_value=Path("/old/schema.json"))
                mock_context.__exit__ = Mock(return_value=None)
                mock_path.return_value = mock_context
                
                result = utils._get_default_schema_path()
                
                assert result == "/old/schema.json"
    
    def test_get_default_schema_path_pkg_resources(self):
        """Test default schema path using pkg_resources fallback"""
        utils = SchemaUtils.__new__(SchemaUtils)
        
        with patch('importlib.resources.files', side_effect=ImportError):
            with patch('pkg_resources.resource_filename', return_value="/pkg/schema.json") as mock_pkg:
                
                result = utils._get_default_schema_path()
                
                assert result == "/pkg/schema.json"
                mock_pkg.assert_called_once_with("signalwire_agents", "schema.json")
    
    def test_get_default_schema_path_manual_search(self):
        """Test default schema path using manual file search"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.log = Mock()
        
        with patch('importlib.resources.files', side_effect=ImportError):
            with patch('pkg_resources.resource_filename', side_effect=ImportError):
                with patch('os.path.exists') as mock_exists:
                    with patch('os.getcwd', return_value="/current"):
                        # First path exists
                        mock_exists.side_effect = lambda path: path == "/current/schema.json"
                        
                        result = utils._get_default_schema_path()
                        
                        assert result == "/current/schema.json"
    
    def test_get_default_schema_path_not_found(self):
        """Test default schema path when file is not found"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.log = Mock()
        
        with patch('importlib.resources.files', side_effect=ImportError):
            with patch('pkg_resources.resource_filename', side_effect=ImportError):
                with patch('os.path.exists', return_value=False):
                    
                    result = utils._get_default_schema_path()
                    
                    assert result is None
    
    def test_load_schema_success(self):
        """Test successful schema loading"""
        schema_data = {
            "$defs": {
                "SWMLMethod": {
                    "anyOf": [{"$ref": "#/$defs/AIMethod"}]
                },
                "AIMethod": {
                    "properties": {
                        "ai": {"type": "object"}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_data, f)
            schema_path = f.name
        
        try:
            utils = SchemaUtils.__new__(SchemaUtils)
            utils.schema_path = schema_path
            utils.log = Mock()
            
            result = utils.load_schema()
            
            assert result == schema_data
        finally:
            os.unlink(schema_path)
    
    def test_load_schema_file_not_found(self):
        """Test schema loading when file doesn't exist"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema_path = "/nonexistent/schema.json"
        utils.log = Mock()
        
        result = utils.load_schema()
        
        assert result == {}
        utils.log.error.assert_called_once()
    
    def test_load_schema_invalid_json(self):
        """Test schema loading with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            schema_path = f.name
        
        try:
            utils = SchemaUtils.__new__(SchemaUtils)
            utils.schema_path = schema_path
            utils.log = Mock()
            
            result = utils.load_schema()
            
            assert result == {}
            utils.log.error.assert_called_once()
        finally:
            os.unlink(schema_path)
    
    def test_load_schema_no_path(self):
        """Test schema loading when no path is provided"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema_path = None
        utils.log = Mock()
        
        result = utils.load_schema()
        
        assert result == {}
        utils.log.warning.assert_called_once()


class TestVerbExtraction:
    """Test verb extraction functionality"""
    
    def test_extract_verb_definitions_success(self):
        """Test successful verb extraction"""
        schema = {
            "$defs": {
                "SWMLMethod": {
                    "anyOf": [
                        {"$ref": "#/$defs/AIMethod"},
                        {"$ref": "#/$defs/AnswerMethod"}
                    ]
                },
                "AIMethod": {
                    "properties": {
                        "ai": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"}
                            }
                        }
                    }
                },
                "AnswerMethod": {
                    "properties": {
                        "answer": {
                            "type": "object",
                            "properties": {
                                "max_duration": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        }
        
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = schema
        utils.log = Mock()
        
        verbs = utils._extract_verb_definitions()
        
        assert "ai" in verbs
        assert "answer" in verbs
        assert verbs["ai"]["name"] == "ai"
        assert verbs["ai"]["schema_name"] == "AIMethod"
        assert verbs["answer"]["name"] == "answer"
        assert verbs["answer"]["schema_name"] == "AnswerMethod"
    
    def test_extract_verb_definitions_no_swml_method(self):
        """Test verb extraction when SWMLMethod is missing"""
        schema = {"$defs": {}}
        
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = schema
        utils.log = Mock()
        
        verbs = utils._extract_verb_definitions()
        
        assert verbs == {}
        utils.log.warning.assert_called_once()
    
    def test_extract_verb_definitions_no_anyof(self):
        """Test verb extraction when anyOf is missing"""
        schema = {
            "$defs": {
                "SWMLMethod": {
                    "properties": {}
                }
            }
        }
        
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = schema
        utils.log = Mock()
        
        verbs = utils._extract_verb_definitions()
        
        assert verbs == {}


class TestVerbProperties:
    """Test verb property access methods"""
    
    def setup_method(self):
        """Set up test data"""
        self.utils = SchemaUtils.__new__(SchemaUtils)
        self.utils.verbs = {
            "ai": {
                "name": "ai",
                "schema_name": "AIMethod",
                "definition": {
                    "properties": {
                        "ai": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "temperature": {"type": "number"}
                            },
                            "required": ["prompt"]
                        }
                    }
                }
            },
            "answer": {
                "name": "answer",
                "schema_name": "AnswerMethod",
                "definition": {
                    "properties": {
                        "answer": {
                            "type": "object",
                            "properties": {
                                "max_duration": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        }
    
    def test_get_verb_properties_existing(self):
        """Test getting properties for existing verb"""
        result = self.utils.get_verb_properties("ai")
        
        expected = {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "temperature": {"type": "number"}
            },
            "required": ["prompt"]
        }
        assert result == expected
    
    def test_get_verb_properties_nonexistent(self):
        """Test getting properties for nonexistent verb"""
        result = self.utils.get_verb_properties("nonexistent")
        
        assert result == {}
    
    def test_get_verb_required_properties_existing(self):
        """Test getting required properties for existing verb"""
        result = self.utils.get_verb_required_properties("ai")
        
        assert result == ["prompt"]
    
    def test_get_verb_required_properties_no_required(self):
        """Test getting required properties when none are specified"""
        result = self.utils.get_verb_required_properties("answer")
        
        assert result == []
    
    def test_get_verb_required_properties_nonexistent(self):
        """Test getting required properties for nonexistent verb"""
        result = self.utils.get_verb_required_properties("nonexistent")
        
        assert result == []
    
    def test_get_all_verb_names(self):
        """Test getting all verb names"""
        result = self.utils.get_all_verb_names()
        
        assert set(result) == {"ai", "answer"}
    
    def test_get_verb_parameters_existing(self):
        """Test getting parameters for existing verb"""
        result = self.utils.get_verb_parameters("ai")
        
        expected = {
            "prompt": {"type": "string"},
            "temperature": {"type": "number"}
        }
        assert result == expected
    
    def test_get_verb_parameters_nonexistent(self):
        """Test getting parameters for nonexistent verb"""
        result = self.utils.get_verb_parameters("nonexistent")
        
        assert result == {}


class TestVerbValidation:
    """Test verb validation functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.utils = SchemaUtils.__new__(SchemaUtils)
        self.utils.verbs = {
            "ai": {
                "name": "ai",
                "schema_name": "AIMethod",
                "definition": {
                    "properties": {
                        "ai": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "temperature": {"type": "number"}
                            },
                            "required": ["prompt"]
                        }
                    }
                }
            }
        }
    
    def test_validate_verb_valid_config(self):
        """Test validation with valid configuration"""
        config = {"prompt": "You are helpful"}
        
        is_valid, errors = self.utils.validate_verb("ai", config)
        
        assert is_valid is True
        assert errors == []
    
    def test_validate_verb_missing_required(self):
        """Test validation with missing required property"""
        config = {"temperature": 0.7}
        
        is_valid, errors = self.utils.validate_verb("ai", config)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "Missing required property 'prompt'" in errors[0]
    
    def test_validate_verb_nonexistent_verb(self):
        """Test validation with nonexistent verb"""
        config = {"some": "config"}
        
        is_valid, errors = self.utils.validate_verb("nonexistent", config)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "Unknown verb: nonexistent" in errors[0]
    
    def test_validate_verb_extra_properties_allowed(self):
        """Test validation allows extra properties"""
        config = {"prompt": "You are helpful", "extra_prop": "value"}
        
        is_valid, errors = self.utils.validate_verb("ai", config)
        
        assert is_valid is True
        assert errors == []


class TestCodeGeneration:
    """Test code generation functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.utils = SchemaUtils.__new__(SchemaUtils)
        self.utils.verbs = {
            "ai": {
                "name": "ai",
                "schema_name": "AIMethod",
                "definition": {
                    "properties": {
                        "ai": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The AI prompt text"
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Temperature for AI generation"
                                }
                            },
                            "required": ["prompt"]
                        }
                    }
                }
            }
        }
    
    def test_generate_method_signature(self):
        """Test method signature generation"""
        result = self.utils.generate_method_signature("ai")
        
        assert "def ai(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> bool:" in result
        assert "Add the ai verb to the current document" in result
        assert "prompt: The AI prompt text" in result
        assert "temperature: Temperature for AI generation" in result
    
    def test_generate_method_body(self):
        """Test method body generation"""
        result = self.utils.generate_method_body("ai")
        
        assert "config = {}" in result
        assert "if prompt is not None:" in result
        assert "config['prompt'] = prompt" in result
        assert "if temperature is not None:" in result
        assert "config['temperature'] = temperature" in result
        assert "return self.add_verb('ai', config)" in result
    
    def test_get_type_annotation_string(self):
        """Test type annotation for string"""
        param_def = {"type": "string"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "str"
    
    def test_get_type_annotation_integer(self):
        """Test type annotation for integer"""
        param_def = {"type": "integer"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "int"
    
    def test_get_type_annotation_number(self):
        """Test type annotation for number"""
        param_def = {"type": "number"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "float"
    
    def test_get_type_annotation_boolean(self):
        """Test type annotation for boolean"""
        param_def = {"type": "boolean"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "bool"
    
    def test_get_type_annotation_array(self):
        """Test type annotation for array"""
        param_def = {
            "type": "array",
            "items": {"type": "string"}
        }
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "List[str]"
    
    def test_get_type_annotation_array_no_items(self):
        """Test type annotation for array without items"""
        param_def = {"type": "array"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "List[Any]"
    
    def test_get_type_annotation_object(self):
        """Test type annotation for object"""
        param_def = {"type": "object"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "Dict[str, Any]"
    
    def test_get_type_annotation_anyof(self):
        """Test type annotation for anyOf"""
        param_def = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "Any"
    
    def test_get_type_annotation_ref(self):
        """Test type annotation for $ref"""
        param_def = {"$ref": "#/$defs/SomeType"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "Any"
    
    def test_get_type_annotation_unknown(self):
        """Test type annotation for unknown type"""
        param_def = {"type": "unknown"}
        
        result = self.utils._get_type_annotation(param_def)
        
        assert result == "Any"


class TestSchemaUtilsIntegration:
    """Test integration scenarios"""
    
    def test_complete_workflow(self):
        """Test complete schema utils workflow"""
        schema_data = {
            "$defs": {
                "SWMLMethod": {
                    "anyOf": [
                        {"$ref": "#/$defs/AIMethod"},
                        {"$ref": "#/$defs/PlayMethod"}
                    ]
                },
                "AIMethod": {
                    "properties": {
                        "ai": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "AI prompt"
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Generation temperature"
                                }
                            },
                            "required": ["prompt"]
                        }
                    }
                },
                "PlayMethod": {
                    "properties": {
                        "play": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "URL to play"
                                }
                            },
                            "required": ["url"]
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_data, f)
            schema_path = f.name
        
        try:
            # Initialize with schema
            utils = SchemaUtils(schema_path)
            
            # Test verb discovery
            verb_names = utils.get_all_verb_names()
            assert "ai" in verb_names
            assert "play" in verb_names
            
            # Test validation
            valid_config = {"prompt": "Hello"}
            is_valid, errors = utils.validate_verb("ai", valid_config)
            assert is_valid is True
            
            invalid_config = {}
            is_valid, errors = utils.validate_verb("ai", invalid_config)
            assert is_valid is False
            
            # Test code generation
            signature = utils.generate_method_signature("ai")
            assert "def ai(" in signature
            
            body = utils.generate_method_body("ai")
            assert "add_verb('ai'" in body
            
        finally:
            os.unlink(schema_path)
    
    def test_error_recovery(self):
        """Test error recovery scenarios"""
        # Test with invalid schema path
        utils = SchemaUtils("/nonexistent/schema.json")
        
        # Should still work with empty schema
        assert utils.get_all_verb_names() == []
        
        # Validation should fail gracefully
        is_valid, errors = utils.validate_verb("ai", {})
        assert is_valid is False
        assert "Unknown verb" in errors[0]
    
    def test_empty_schema_handling(self):
        """Test handling of empty schema"""
        with patch.object(SchemaUtils, 'load_schema', return_value={}):
            utils = SchemaUtils("/path/to/schema.json")

            assert utils.get_all_verb_names() == []
            assert utils.get_verb_properties("ai") == {}
            assert utils.get_verb_parameters("ai") == {}

            is_valid, errors = utils.validate_verb("ai", {})
            assert is_valid is False


class TestCompositionKeywords:
    """Test handling of allOf, anyOf, and oneOf composition keywords"""

    def test_resolve_ref_valid(self):
        """Test resolving a valid $ref"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "MyType": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
        utils.log = Mock()

        result = utils._resolve_ref("#/$defs/MyType")

        assert result == {"type": "object", "properties": {"name": {"type": "string"}}}

    def test_resolve_ref_not_found(self):
        """Test resolving a $ref that doesn't exist"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {"$defs": {}}
        utils.log = Mock()

        result = utils._resolve_ref("#/$defs/NonExistent")

        assert result == {}

    def test_resolve_ref_invalid_format(self):
        """Test resolving a $ref with unsupported format"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {"$defs": {}}
        utils.log = Mock()

        result = utils._resolve_ref("http://example.com/schema")

        assert result == {}

    def test_merge_properties_from_allof(self):
        """Test merging properties from allOf - all required fields should be merged"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "BaseType": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"]
                },
                "ExtendedType": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"]
                }
            }
        }
        utils.log = Mock()

        composition = [
            {"$ref": "#/$defs/BaseType"},
            {"$ref": "#/$defs/ExtendedType"}
        ]

        result = utils._merge_properties_from_composition(composition, "allOf")

        assert "properties" in result
        assert "id" in result["properties"]
        assert "name" in result["properties"]
        assert "required" in result
        assert set(result["required"]) == {"id", "name"}

    def test_merge_properties_from_oneof(self):
        """Test merging properties from oneOf - required fields should NOT be merged"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "TypeA": {
                    "type": "object",
                    "properties": {"a_prop": {"type": "string"}},
                    "required": ["a_prop"]
                },
                "TypeB": {
                    "type": "object",
                    "properties": {"b_prop": {"type": "integer"}},
                    "required": ["b_prop"]
                }
            }
        }
        utils.log = Mock()

        composition = [
            {"$ref": "#/$defs/TypeA"},
            {"$ref": "#/$defs/TypeB"}
        ]

        result = utils._merge_properties_from_composition(composition, "oneOf")

        # Properties are merged (for discovery)
        assert "properties" in result
        assert "a_prop" in result["properties"]
        assert "b_prop" in result["properties"]
        # Required fields are NOT merged for oneOf
        assert "required" not in result

    def test_merge_properties_from_anyof(self):
        """Test merging properties from anyOf - required fields should NOT be merged"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "TypeA": {
                    "type": "object",
                    "properties": {"a_prop": {"type": "string"}},
                    "required": ["a_prop"]
                },
                "TypeB": {
                    "type": "object",
                    "properties": {"b_prop": {"type": "integer"}},
                    "required": ["b_prop"]
                }
            }
        }
        utils.log = Mock()

        composition = [
            {"$ref": "#/$defs/TypeA"},
            {"$ref": "#/$defs/TypeB"}
        ]

        result = utils._merge_properties_from_composition(composition, "anyOf")

        # Properties are merged (for discovery)
        assert "properties" in result
        assert "a_prop" in result["properties"]
        assert "b_prop" in result["properties"]
        # Required fields are NOT merged for anyOf
        assert "required" not in result

    def test_merge_properties_inline_definitions(self):
        """Test merging properties from inline definitions (not $ref)"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {"$defs": {}}
        utils.log = Mock()

        composition = [
            {"type": "object", "properties": {"inline_a": {"type": "string"}}},
            {"type": "object", "properties": {"inline_b": {"type": "integer"}}}
        ]

        result = utils._merge_properties_from_composition(composition, "anyOf")

        assert "properties" in result
        assert "inline_a" in result["properties"]
        assert "inline_b" in result["properties"]

    def test_extract_verb_definitions_with_oneof(self):
        """Test verb extraction when SWMLMethod uses oneOf"""
        schema = {
            "$defs": {
                "SWMLMethod": {
                    "oneOf": [
                        {"$ref": "#/$defs/PlayMethod"},
                        {"$ref": "#/$defs/StopMethod"}
                    ]
                },
                "PlayMethod": {
                    "properties": {
                        "play": {"type": "object", "properties": {"url": {"type": "string"}}}
                    }
                },
                "StopMethod": {
                    "properties": {
                        "stop": {"type": "object", "properties": {}}
                    }
                }
            }
        }

        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = schema
        utils.log = Mock()

        verbs = utils._extract_verb_definitions()

        assert "play" in verbs
        assert "stop" in verbs

    def test_extract_verb_definitions_with_allof(self):
        """Test verb extraction when SWMLMethod uses allOf"""
        schema = {
            "$defs": {
                "SWMLMethod": {
                    "allOf": [
                        {"$ref": "#/$defs/RecordMethod"}
                    ]
                },
                "RecordMethod": {
                    "properties": {
                        "record": {"type": "object", "properties": {"format": {"type": "string"}}}
                    }
                }
            }
        }

        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = schema
        utils.log = Mock()

        verbs = utils._extract_verb_definitions()

        assert "record" in verbs

    def test_get_verb_properties_with_oneof(self):
        """Test get_verb_properties when verb value uses oneOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "ConnectSingle": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"]
                },
                "ConnectMultiple": {
                    "type": "object",
                    "properties": {"to": {"type": "array"}, "timeout": {"type": "integer"}}
                }
            }
        }
        utils.verbs = {
            "connect": {
                "name": "connect",
                "schema_name": "Connect",
                "definition": {
                    "properties": {
                        "connect": {
                            "oneOf": [
                                {"$ref": "#/$defs/ConnectSingle"},
                                {"$ref": "#/$defs/ConnectMultiple"}
                            ],
                            "description": "Connect to endpoint"
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_properties("connect")

        # Should have merged properties
        assert "properties" in result
        assert "to" in result["properties"]
        assert "timeout" in result["properties"]
        # Should preserve description
        assert result.get("description") == "Connect to endpoint"

    def test_get_verb_properties_with_anyof(self):
        """Test get_verb_properties when verb value uses anyOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "PlayURL": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}}
                },
                "PlayTTS": {
                    "type": "object",
                    "properties": {"say": {"type": "string"}, "voice": {"type": "string"}}
                }
            }
        }
        utils.verbs = {
            "play": {
                "name": "play",
                "schema_name": "Play",
                "definition": {
                    "properties": {
                        "play": {
                            "anyOf": [
                                {"$ref": "#/$defs/PlayURL"},
                                {"$ref": "#/$defs/PlayTTS"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_properties("play")

        assert "properties" in result
        assert "url" in result["properties"]
        assert "say" in result["properties"]
        assert "voice" in result["properties"]

    def test_get_verb_properties_with_allof(self):
        """Test get_verb_properties when verb value uses allOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "BaseConfig": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"]
                },
                "ExtendedConfig": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"]
                }
            }
        }
        utils.verbs = {
            "config": {
                "name": "config",
                "schema_name": "Config",
                "definition": {
                    "properties": {
                        "config": {
                            "allOf": [
                                {"$ref": "#/$defs/BaseConfig"},
                                {"$ref": "#/$defs/ExtendedConfig"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_properties("config")

        assert "properties" in result
        assert "enabled" in result["properties"]
        assert "level" in result["properties"]

    def test_get_verb_required_properties_with_allof(self):
        """Test get_verb_required_properties when verb value uses allOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "BaseConfig": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"]
                },
                "ExtendedConfig": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"]
                }
            }
        }
        utils.verbs = {
            "config": {
                "name": "config",
                "schema_name": "Config",
                "definition": {
                    "properties": {
                        "config": {
                            "allOf": [
                                {"$ref": "#/$defs/BaseConfig"},
                                {"$ref": "#/$defs/ExtendedConfig"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_required_properties("config")

        # allOf should merge all required fields
        assert set(result) == {"enabled", "level"}

    def test_get_verb_required_properties_with_oneof(self):
        """Test get_verb_required_properties when verb value uses oneOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "TypeA": {
                    "type": "object",
                    "properties": {"a_prop": {"type": "string"}},
                    "required": ["a_prop"]
                },
                "TypeB": {
                    "type": "object",
                    "properties": {"b_prop": {"type": "string"}},
                    "required": ["b_prop"]
                }
            }
        }
        utils.verbs = {
            "test": {
                "name": "test",
                "schema_name": "Test",
                "definition": {
                    "properties": {
                        "test": {
                            "oneOf": [
                                {"$ref": "#/$defs/TypeA"},
                                {"$ref": "#/$defs/TypeB"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_required_properties("test")

        # oneOf should NOT merge required fields
        assert result == []

    def test_get_verb_parameters_with_composition(self):
        """Test get_verb_parameters works correctly with composition keywords"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "ConnectSingle": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}}
                },
                "ConnectMultiple": {
                    "type": "object",
                    "properties": {"timeout": {"type": "integer"}}
                }
            }
        }
        utils.verbs = {
            "connect": {
                "name": "connect",
                "schema_name": "Connect",
                "definition": {
                    "properties": {
                        "connect": {
                            "oneOf": [
                                {"$ref": "#/$defs/ConnectSingle"},
                                {"$ref": "#/$defs/ConnectMultiple"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_parameters("connect")

        assert "to" in result
        assert "timeout" in result

    def test_get_type_annotation_allof(self):
        """Test type annotation for allOf returns Any"""
        utils = SchemaUtils.__new__(SchemaUtils)

        param_def = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "integer"}}}
            ]
        }

        result = utils._get_type_annotation(param_def)

        assert result == "Any"

    def test_get_type_annotation_oneof(self):
        """Test type annotation for oneOf returns Any"""
        utils = SchemaUtils.__new__(SchemaUtils)

        param_def = {
            "oneOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }

        result = utils._get_type_annotation(param_def)

        assert result == "Any"

    def test_validate_verb_with_composition(self):
        """Test validation works correctly with verbs using composition keywords"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "BaseConfig": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"]
                },
                "ExtendedConfig": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"]
                }
            }
        }
        utils.verbs = {
            "config": {
                "name": "config",
                "schema_name": "Config",
                "definition": {
                    "properties": {
                        "config": {
                            "allOf": [
                                {"$ref": "#/$defs/BaseConfig"},
                                {"$ref": "#/$defs/ExtendedConfig"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        # Valid config with all required fields
        is_valid, errors = utils.validate_verb("config", {"enabled": True, "level": 5})
        assert is_valid is True
        assert errors == []

        # Invalid config missing required fields
        is_valid, errors = utils.validate_verb("config", {"enabled": True})
        assert is_valid is False
        assert any("level" in e for e in errors)

    def test_integration_schema_with_oneof_verbs(self):
        """Integration test for schema with oneOf verb definitions"""
        schema_data = {
            "$defs": {
                "SWMLMethod": {
                    "oneOf": [
                        {"$ref": "#/$defs/Connect"},
                        {"$ref": "#/$defs/Play"}
                    ]
                },
                "Connect": {
                    "properties": {
                        "connect": {
                            "oneOf": [
                                {"$ref": "#/$defs/ConnectSingle"},
                                {"$ref": "#/$defs/ConnectParallel"}
                            ],
                            "description": "Connect to endpoint"
                        }
                    }
                },
                "ConnectSingle": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"]
                },
                "ConnectParallel": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "array"},
                        "timeout": {"type": "integer"}
                    }
                },
                "Play": {
                    "properties": {
                        "play": {
                            "type": "object",
                            "properties": {"url": {"type": "string"}},
                            "required": ["url"]
                        }
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_data, f)
            schema_path = f.name

        try:
            utils = SchemaUtils(schema_path)

            # Verb extraction should work
            verb_names = utils.get_all_verb_names()
            assert "connect" in verb_names
            assert "play" in verb_names

            # Should get merged properties for connect (oneOf)
            connect_params = utils.get_verb_parameters("connect")
            assert "to" in connect_params
            assert "timeout" in connect_params

            # Play should work normally
            play_params = utils.get_verb_parameters("play")
            assert "url" in play_params

            # Validation should work
            is_valid, errors = utils.validate_verb("play", {"url": "http://example.com"})
            assert is_valid is True

        finally:
            os.unlink(schema_path)

    def test_anyof_with_shared_required_fields(self):
        """Test anyOf where branches share common required fields (like SMS scenario)"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "SMSWithBody": {
                    "type": "object",
                    "properties": {
                        "to_number": {"type": "string"},
                        "from_number": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to_number", "from_number", "body"]
                },
                "SMSWithMedia": {
                    "type": "object",
                    "properties": {
                        "to_number": {"type": "string"},
                        "from_number": {"type": "string"},
                        "media": {"type": "array"},
                        "body": {"type": "string"}  # optional here
                    },
                    "required": ["to_number", "from_number", "media"]
                }
            }
        }
        utils.log = Mock()

        composition = [
            {"$ref": "#/$defs/SMSWithBody"},
            {"$ref": "#/$defs/SMSWithMedia"}
        ]

        result = utils._merge_properties_from_composition(composition, "anyOf")

        # All properties should be merged
        assert "properties" in result
        assert "to_number" in result["properties"]
        assert "from_number" in result["properties"]
        assert "body" in result["properties"]
        assert "media" in result["properties"]

        # For anyOf: only fields required in ALL branches should be required
        # Intersection of ["to_number", "from_number", "body"] and ["to_number", "from_number", "media"]
        # = ["to_number", "from_number"]
        assert "required" in result
        assert set(result["required"]) == {"to_number", "from_number"}

    def test_oneof_with_type_conflicts(self):
        """Test oneOf where same property has different types"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "SingleValue": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}}
                },
                "MultiValue": {
                    "type": "object",
                    "properties": {"value": {"type": "array"}}
                }
            }
        }
        utils.log = Mock()

        composition = [
            {"$ref": "#/$defs/SingleValue"},
            {"$ref": "#/$defs/MultiValue"}
        ]

        result = utils._merge_properties_from_composition(composition, "oneOf")

        # Property should be merged but marked as conflicting
        assert "properties" in result
        assert "value" in result["properties"]
        assert result["properties"]["value"].get("_conflicting_types") is True

    def test_type_annotation_for_conflicting_types(self):
        """Test that conflicting types return Any annotation"""
        utils = SchemaUtils.__new__(SchemaUtils)

        # Property with conflicting types flag
        param_def = {
            "type": "string",
            "_conflicting_types": True
        }

        result = utils._get_type_annotation(param_def)

        assert result == "Any"

    def test_type_annotation_for_normal_property(self):
        """Test that normal properties still get correct type annotation"""
        utils = SchemaUtils.__new__(SchemaUtils)

        # Property without conflicting types flag
        param_def = {"type": "string"}

        result = utils._get_type_annotation(param_def)

        assert result == "str"

    def test_verb_required_with_shared_required_anyof(self):
        """Test get_verb_required_properties returns intersection for anyOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "SMSWithBody": {
                    "type": "object",
                    "properties": {
                        "to_number": {"type": "string"},
                        "from_number": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to_number", "from_number", "body"]
                },
                "SMSWithMedia": {
                    "type": "object",
                    "properties": {
                        "to_number": {"type": "string"},
                        "from_number": {"type": "string"},
                        "media": {"type": "array"}
                    },
                    "required": ["to_number", "from_number", "media"]
                }
            }
        }
        utils.verbs = {
            "send_sms": {
                "name": "send_sms",
                "schema_name": "SendSMS",
                "definition": {
                    "properties": {
                        "send_sms": {
                            "anyOf": [
                                {"$ref": "#/$defs/SMSWithBody"},
                                {"$ref": "#/$defs/SMSWithMedia"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        result = utils.get_verb_required_properties("send_sms")

        # Should return intersection: to_number and from_number
        assert set(result) == {"to_number", "from_number"}

    def test_validate_verb_with_shared_required_anyof(self):
        """Test validation correctly requires intersection fields for anyOf"""
        utils = SchemaUtils.__new__(SchemaUtils)
        utils.schema = {
            "$defs": {
                "SMSWithBody": {
                    "type": "object",
                    "properties": {
                        "to_number": {"type": "string"},
                        "from_number": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["to_number", "from_number", "body"]
                },
                "SMSWithMedia": {
                    "type": "object",
                    "properties": {
                        "to_number": {"type": "string"},
                        "from_number": {"type": "string"},
                        "media": {"type": "array"}
                    },
                    "required": ["to_number", "from_number", "media"]
                }
            }
        }
        utils.verbs = {
            "send_sms": {
                "name": "send_sms",
                "schema_name": "SendSMS",
                "definition": {
                    "properties": {
                        "send_sms": {
                            "anyOf": [
                                {"$ref": "#/$defs/SMSWithBody"},
                                {"$ref": "#/$defs/SMSWithMedia"}
                            ]
                        }
                    }
                }
            }
        }
        utils.log = Mock()

        # Valid: has both shared required fields
        is_valid, errors = utils.validate_verb("send_sms", {
            "to_number": "+1234567890",
            "from_number": "+0987654321",
            "body": "Hello"  # one variant's specific field
        })
        assert is_valid is True

        # Invalid: missing shared required field
        is_valid, errors = utils.validate_verb("send_sms", {
            "to_number": "+1234567890",
            "body": "Hello"
        })
        assert is_valid is False
        assert any("from_number" in e for e in errors)