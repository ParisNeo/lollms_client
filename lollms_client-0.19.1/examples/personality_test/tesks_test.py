from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client import TasksLibrary
from ascii_colors import ASCIIColors

lc = LollmsClient("http://localhost:9600")
tl = TasksLibrary(lc)

# ======================================= Multichoice Q&A ==========================
# Define a multichoice question
question = "What is the capital city of France?"

# Define the possible answers
possible_answers = ["Paris", "Berlin", "London", "Madrid"]

# Call the multichoice_question function with the question and possible answers
selected_option = tl.multichoice_question(question, possible_answers)


ASCIIColors.yellow(question)
ASCIIColors.green(possible_answers[selected_option])

# ======================================= Yes no  ==========================
# Define a yes or no question
question = "Is Paris the capital city of France?"

# Call the yes_no function with the question
answer = tl.yes_no(question)
ASCIIColors.yellow(question)
ASCIIColors.green("Yes" if answer else "No")


# ======================================= Code extraction  ==========================
# Define a text with code blocks
text = """
Here is some text with a code block:
```python
def hello_world():
    print("Hello, world!")
```
And here is another code block:
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```
"""

# Call the extract_code_blocks function with the text
code_blocks = tl.extract_code_blocks(text)

# Print the extracted code blocks
for i, code_block in enumerate(code_blocks):
    ASCIIColors.bold(f"Code block {i + 1}:")
    ASCIIColors.bold(f"Index: {code_block['index']}")
    ASCIIColors.bold(f"File name: {code_block['file_name']}")
    ASCIIColors.bold(f"Content: {code_block['content']}")
    ASCIIColors.bold(f"Type: {code_block['type']}")
    print()


