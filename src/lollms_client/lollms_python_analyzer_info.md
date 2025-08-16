# Information for lollms_python_analyzer.py

File: `lollms_client\lollms_python_analyzer.py`

## Classes

### MethodInfo

```python
class MethodInfo:
    def __init__(self, node: ast.FunctionDef) -> Any
    def _parse_args(self, node: ast.FunctionDef) -> List[str]
    def _get_type_annotation(self, annotation: Optional[ast.AST]) -> str
    def __str__(self) -> str
```

### ClassInfo

```python
class ClassInfo:
    def __init__(self, node: ast.ClassDef) -> Any
    def _parse_methods(self, node: ast.ClassDef) -> List[MethodInfo]
    def __str__(self) -> str
```

### EnumInfo

```python
class EnumInfo:
    def __init__(self, node: ast.ClassDef) -> Any
    def _parse_members(self, node: ast.ClassDef) -> List[str]
    def __str__(self) -> str
```

### FunctionInfo

```python
class FunctionInfo:
    def __init__(self, node: ast.FunctionDef) -> Any
    def _parse_args(self, node: ast.FunctionDef) -> List[str]
    def _get_type_annotation(self, annotation: Optional[ast.AST]) -> str
    def __str__(self) -> str
```

### Analyzer

```python
class Analyzer:
    def __init__(self, file_path: Path) -> Any
    def analyze(self) -> None
    def _parse_file(self) -> ast.AST
    def _analyze_classes(self, tree: ast.AST) -> List[ClassInfo]
    def _analyze_enums(self, tree: ast.AST) -> List[EnumInfo]
    def _analyze_functions(self, tree: ast.AST) -> List[FunctionInfo]
    def _is_enum(self, node: ast.ClassDef) -> bool
    def generate_markdown(self) -> str
    def save_markdown(self, output_path: Optional[Path] = None) -> None
    def generate_documentation(self) -> None
```

## Functions

### main

```python
def main(file_path: Path) -> None
```

