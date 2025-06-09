## DOCS FOR: `lollms_client/lollms_config.py` (Configuration Utilities)

**Purpose:**
This module provides utility classes for managing configurations, particularly for defining configuration templates and handling typed configurations that can be loaded from and saved to YAML files.

---
### `ConfigTemplate`

*   **Purpose**: Represents a template for configuration settings. It allows defining entries with names, default values, types, and optional constraints (min/max) and help text. This is useful for generating UI for settings or for validating configurations.
*   **Key Attributes**:
    *   `template` (List[Dict]]): A list of dictionaries, where each dictionary defines a configuration entry. Each entry dictionary typically contains:
        *   `"name"` (str): The unique name/key of the setting.
        *   `"value"` (Any): The default value for the setting.
        *   `"type"` (str): The expected data type (e.g., "int", "float", "str", "bool", "list", "dict").
        *   `"min"` (Optional[Any]): Minimum allowed value (for numerical types).
        *   `"max"` (Optional[Any]): Maximum allowed value (for numerical types).
        *   `"options"` (Optional[List[Any]]): A list of valid options for string types (enum-like).
        *   `"help"` (str, optional): A description of the setting.
*   **Methods**:
    *   **`__init__(template: List[Dict] = None)`**:
        *   **Purpose**: Initializes the template.
        *   **Parameters**: `template` (Optional[List[Dict]]): An initial list of configuration entry dictionaries.
    *   **`add_entry(entry_name: str, entry_value: Any, entry_type: str, entry_min: Optional[Any] = None, entry_max: Optional[Any] = None, entry_help: str = '')`**:
        *   **Purpose**: Adds a new configuration entry to the template.
        *   **Parameters**: As described in Key Attributes.
    *   **`__getitem__(key: str) -> Dict`**:
        *   **Purpose**: Retrieves the dictionary defining the configuration entry for the given `key` (entry name).
        *   **Returns**: The entry dictionary.
    *   **`__getattr__(key: str) -> Any`**:
        *   **Purpose**: Retrieves the *value* of the configuration entry with the specified `key`.
        *   **Returns**: The value of the entry.
    *   **`__setattr__(key: str, value: Any)`**:
        *   **Purpose**: Sets the *value* of the configuration entry with the specified `key`.
    *   **`__setitem__(key: str, value: Any)`**:
        *   **Purpose**: Sets the *value* of the configuration entry with the specified `key`. (Same as `__setattr__` for value modification).
    *   **`__contains__(item: str) -> bool`**:
        *   **Purpose**: Checks if a configuration entry with the specified `item` name exists in the template.

---
### `BaseConfig`

*   **Purpose**: A base class for managing configuration data loaded from/saved to YAML files. It provides dictionary-like access to configuration values.
*   **Key Attributes**:
    *   `config` (Dict): The actual configuration data.
    *   `file_path` (Optional[Path]): The path to the YAML file this configuration is associated with.
    *   `exceptional_keys` (List[str]): A list of keys that are treated as direct attributes of the `BaseConfig` instance itself, rather than keys within the `config` dictionary.
*   **Methods**:
    *   **`__init__(exceptional_keys: List[str] = [], config: Optional[Dict] = None, file_path: Optional[Union[Path, str]] = None)`**:
        *   **Purpose**: Initializes the configuration.
        *   **Parameters**:
            *   `exceptional_keys`: Keys to be handled as instance attributes.
            *   `config`: An initial dictionary of configuration data.
            *   `file_path`: Path to an associated YAML file (can be loaded via `load_config`).
    *   **`from_template(template: ConfigTemplate, exceptional_keys: List[str] = [], file_path: Optional[Union[Path, str]] = None) -> 'BaseConfig'` (staticmethod)**:
        *   **Purpose**: Creates a `BaseConfig` instance from a `ConfigTemplate`, using the default values from the template.
        *   **Parameters**: `template`, `exceptional_keys`, `file_path`.
        *   **Returns**: A new `BaseConfig` instance.
    *   **`to_dict() -> Dict`**:
        *   **Purpose**: Returns the internal configuration data as a dictionary.
    *   **`__getitem__(key: Any) -> Any`**:
        *   **Purpose**: Accesses a configuration value by key.
    *   **`copy() -> 'BaseConfig'`**:
        *   **Purpose**: Creates a deep copy of the configuration.
    *   **`__getattr__(key: str) -> Any`**:
        *   **Purpose**: Accesses a configuration value as an attribute (unless it's an exceptional key).
    *   **`__setattr__(key: str, value: Any)`**:
        *   **Purpose**: Sets a configuration value. If `key` is in `exceptional_keys` or `config` is not yet initialized, it sets it as a direct instance attribute. Otherwise, it sets it within the `config` dictionary.
    *   **`__setitem__(key: str, value: Any)`**:
        *   **Purpose**: Sets a configuration value by key within the `config` dictionary.
    *   **`__contains__(item: str) -> bool`**:
        *   **Purpose**: Checks if a key exists in the `config` dictionary.
    *   **`load_config(file_path: Optional[Union[Path, str]] = None)`**:
        *   **Purpose**: Loads configuration data from a YAML file into the `config` attribute.
        *   **Parameters**: `file_path` (Optional): If provided, updates the instance's `file_path` and loads from there. Otherwise, uses the existing `file_path`.
    *   **`save_config(file_path: Optional[Union[Path, str]] = None)`**:
        *   **Purpose**: Saves the current `config` data to a YAML file.
        *   **Parameters**: `file_path` (Optional): If provided, updates the instance's `file_path` and saves there. Otherwise, uses the existing `file_path`.

---
### `TypedConfig`

*   **Purpose**: Extends `BaseConfig` by associating it with a `ConfigTemplate`. This allows for type validation and easier management of configurations that adhere to a predefined structure.
*   **Key Attributes**:
    *   `config_template` (ConfigTemplate): The template defining the structure and types of the configuration.
    *   `config` (BaseConfig): The underlying `BaseConfig` object holding the actual values.
*   **Methods**:
    *   **`__init__(config_template: ConfigTemplate, config: BaseConfig)`**:
        *   **Purpose**: Initializes a `TypedConfig` instance.
        *   **Parameters**: `config_template`, `config`.
    *   **`addConfigs(cfg_template: List[Dict])`**:
        *   **Purpose**: Adds new entries to the internal `config_template`.
    *   **`update_template(new_template: ConfigTemplate)`**:
        *   **Purpose**: Replaces the existing `config_template` with a new one and re-synchronizes values.
    *   **`get(key: str, default_value: Optional[Any] = None) -> Any`**:
        *   **Purpose**: Retrieves a configuration value by key from the underlying `config`.
        *   **Returns**: The value, or `default_value` if the key is not found.
    *   **`__getattr__(key: str) -> Any`**:
        *   **Purpose**: Accesses a configuration value from the template's default or the `config` object.
    *   **`__setattr__(key: str, value: Any)`**:
        *   **Purpose**: Sets a configuration value in both the template (if the key exists) and the underlying `config` object. Ensures type consistency based on the template.
    *   **`__getitem__(key: str) -> Any`**: (Same as `__getattr__` for value retrieval)
    *   **`__setitem__(key: str, value: Any)`**: (Same as `__setattr__` for value setting)
    *   **`sync()`**:
        *   **Purpose**: Updates the values in the `config_template` based on the values currently in the `config` (BaseConfig) object. This is useful if the `config` object was modified directly.
    *   **`set_config(config: BaseConfig)`**:
        *   **Purpose**: Replaces the internal `config` object with a new `BaseConfig` instance and then calls `sync()`.
    *   **`save(file_path: Optional[Union[str, Path]] = None)`**:
        *   **Purpose**: Saves the underlying `BaseConfig` to a YAML file.
    *   **`to_dict(use_template: bool = False) -> Dict`**:
        *   **Purpose**: Returns the configuration as a dictionary.
        *   **Parameters**: `use_template` (bool): If `True`, returns a dictionary derived from the `config_template` (name-value pairs). If `False` (default), returns the `to_dict()` of the underlying `BaseConfig`.

**Usage Example (Conceptual):**
```python
from lollms_client.lollms_config import ConfigTemplate, BaseConfig, TypedConfig
from pathlib import Path

# 1. Define a ConfigTemplate
template = ConfigTemplate()
template.add_entry("model_name", "default_model", "str", entry_help="Name of the LLM model.")
template.add_entry("temperature", 0.7, "float", entry_min=0.0, entry_max=2.0, entry_help="Sampling temperature.")
template.add_entry("use_gpu", True, "bool", entry_help="Whether to use GPU.")

# 2. Create a BaseConfig, perhaps from the template or loading from a file
# config_file = Path("my_settings.yaml")
# base_cfg = BaseConfig(file_path=config_file)
# if config_file.exists():
#     base_cfg.load_config()
# else: # Initialize from template if file doesn't exist
base_cfg = BaseConfig.from_template(template)
base_cfg["model_name"] = "custom_model_from_code" # Override a value

# 3. Create a TypedConfig
typed_configuration = TypedConfig(config_template=template, config=base_cfg)

# Accessing and modifying values
print(f"Model Name: {typed_configuration.model_name}") # Access via getattr
typed_configuration.temperature = 0.85                # Set via setattr

# typed_configuration["use_gpu"] = False              # Set via setitem (works similarly)
# print(f"Use GPU: {typed_configuration['use_gpu']}") # Access via getitem

# Save the configuration
# typed_configuration.save("my_updated_settings.yaml")

# Print all settings
print("\nCurrent TypedConfig:")
for entry in typed_configuration.config_template.template:
    print(f"  {entry['name']}: {typed_configuration[entry['name']]} (Type: {entry['type']})")

print("\nBaseConfig dictionary:")
print(typed_configuration.config.to_dict())

print("\nTypedConfig as dict (from template):")
print(typed_configuration.to_dict(use_template=True))
```

**Dependencies:**
*   `pyyaml` (for loading/saving YAML files in `BaseConfig`)
*   Standard Python libraries (`enum`, `pathlib`, `typing`).