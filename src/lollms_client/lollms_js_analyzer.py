from pathlib import Path
from typing import List, Optional

class MethodInfo:
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

class ClassInfo:
    def __init__(self, name: str, methods: List[MethodInfo]):
        self.name = name
        self.methods = methods

    def __str__(self) -> str:
        return f"class {self.name}:\n" + "\n".join(f"    {method}" for method in self.methods)

class FunctionInfo:
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args

    def __str__(self) -> str:
        return f"function {self.name}({', '.join(self.args)})"

class Analyzer:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.classes: List[ClassInfo] = []
        self.functions: List[FunctionInfo] = []

    def analyze(self) -> None:
        js_code = self._parse_file()
        self._extract_classes_and_functions(js_code)

    def _parse_file(self) -> List[str]:
        with self.file_path.open('r') as file:
            return file.readlines()

    def _extract_classes_and_functions(self, js_code: List[str]) -> None:
        class_name = None

        for line in js_code:
            stripped_line = line.strip()

            # Check for class definition
            if stripped_line.startswith("class "):
                if class_name:  # If we were already in a class, we need to close it
                    self.classes.append(ClassInfo(class_name, []))
                class_name = stripped_line.split()[1].split('{')[0]  # Get class name
                self._parse_class_methods(js_code, class_name)

            # Check for standalone functions
            if not class_name and stripped_line.startswith("function "):
                function_name = stripped_line.split()[1].split('(')[0]
                function_args = self._extract_function_args(js_code, stripped_line)
                self.functions.append(FunctionInfo(function_name, function_args))

            # Check for async functions
            if not class_name and stripped_line.startswith("async function "):
                function_name = stripped_line.split()[2].split('(')[0]
                function_args = self._extract_function_args(js_code, stripped_line)
                self.functions.append(FunctionInfo(f"async {function_name}", function_args))

    def _parse_class_methods(self, js_code: List[str], class_name: str) -> None:
        brace_count = 0
        method_name = None
        method_args = []
        in_multiline = False

        for line in js_code:
            stripped_line = line.strip()

            if stripped_line.startswith("class ") and method_name is None:
                # We've reached the next class definition, exit the method parsing
                break

            if brace_count == 0 and ("constructor" in stripped_line or stripped_line.endswith("{")):
                brace_count += 1  # Increment for class constructor or method start
                if "constructor" in stripped_line:
                    method_name = "constructor"
                    method_args = self._extract_function_args(js_code, stripped_line)
                else:
                    # Check for async methods
                    if stripped_line.startswith("async "):
                        method_name = stripped_line.split()[1].split('(')[0]
                        method_args = self._extract_function_args(js_code, stripped_line)
                        method_name = f"async {method_name}"
                    else:
                        method_name = stripped_line.split()[0]
                        method_args = self._extract_function_args(js_code, stripped_line)

            if brace_count > 0:
                if "{" in stripped_line:
                    brace_count += stripped_line.count("{")
                if "}" in stripped_line:
                    brace_count -= stripped_line.count("}")

                if brace_count == 0 and method_name:
                    # Only append if there is a class to append to
                    if self.classes:
                        self.classes[-1].methods.append(MethodInfo(method_name, method_args))
                    method_name = None
                    method_args = []

            # Handle multi-line function definitions
            if brace_count > 0 and method_name:
                if stripped_line.endswith("}") and in_multiline:
                    # End of method body
                    brace_count -= 1
                    if brace_count == 0:
                        in_multiline = False
                        if self.classes:
                            self.classes[-1].methods.append(MethodInfo(method_name, method_args))
                        method_name = None
                        method_args = []
                elif stripped_line.endswith("{"):
                    # Start of method body
                    in_multiline = True

    def _extract_function_args(self, js_code: List[str], line: str) -> List[str]:
        if '(' in line and ')' in line:
            args = line.split('(')[1].split(')')[0]
            brace_count = 0
            full_args = args.strip()

            # Handle brace counting to capture full argument list
            for char in full_args:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                if brace_count < 0:
                    break

            # If brace_count is not zero, we need to read more lines
            if brace_count > 0:
                for next_line in js_code:
                    next_stripped_line = next_line.strip()
                    full_args += ' ' + next_stripped_line
                    brace_count += next_stripped_line.count('{')
                    brace_count -= next_stripped_line.count('}')
                    if brace_count == 0:
                        break

            # Handle default parameters and destructuring
            full_args = full_args.replace('=', ':')  # Replace '=' with ':' for easier parsing
            return [arg.strip() for arg in full_args.split(',') if arg.strip()]
        return []

    def generate_markdown(self) -> str:
        markdown = f"# Information for {self.file_path.name}\n\n"
        markdown += f"File: `{self.file_path}`\n\n"

        if self.classes:
            markdown += "## Classes\n\n"
            for class_info in self.classes:
                markdown += f"### {class_info.name}\n\n"
                markdown += "```javascript\n"
                markdown += str(class_info) + "\n"
                markdown += "```\n\n"

        if self.functions:
            markdown += "## Functions\n\n"
            for function_info in self.functions:
                markdown += f"### {function_info.name}\n\n"
                markdown += "```javascript\n"
                markdown += str(function_info) + "\n"
                markdown += "```\n\n"

        return markdown

    def save_markdown(self, output_path: Optional[Path] = None) -> None:
        if output_path is None:
            output_path = self.file_path.with_name(f"{self.file_path.stem}_info.md")

        markdown = self.generate_markdown()
        output_path.write_text(markdown)
        print(f"Markdown file generated: {output_path}")

def main(file_path: Path) -> None:
    analyzer = Analyzer(file_path)
    analyzer.analyze()
    analyzer.save_markdown()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse JavaScript file and extract class and function information.")
    parser.add_argument("file", nargs="?", default=r"E:\test2_lollms\lollms_client_js\lollms_client_js\lollms_client_js.js", type=Path,
                        help="Path to the JavaScript file to analyze (default: example.js)")
    args = parser.parse_args()

    file_path = args.file
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        exit(1)

    main(file_path)
