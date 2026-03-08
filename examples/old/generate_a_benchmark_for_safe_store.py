from lollms_client import LollmsClient
from ascii_colors import ASCIIColors, trace_exception, ProgressBar
import pipmaster as pm
pm.ensure_packages(["datasets"])
#assuming you have an active lollms_webui instance running
#you can also use ollama or openai or any other lollmc_client binding 
lc = LollmsClient()

from datasets import load_dataset
import json
# 1. Define the dataset name
dataset_name = "agentlans/high-quality-english-sentences"

try:
    # 2. Load the dataset
    # This dataset only has a 'train' split by default.
    # If a dataset had multiple splits (e.g., 'train', 'validation', 'test'),
    # load_dataset() would return a DatasetDict.
    # We can directly access the 'train' split.
    dataset = load_dataset(dataset_name, split='train')
    print(f"Dataset loaded successfully: {dataset_name}")
    print(f"Dataset structure: {dataset}")

    # 3. Extract the sentences into a list
    # The sentences are in a column likely named 'text' (common for text datasets).
    # Let's inspect the features to be sure.
    print(f"Dataset features: {dataset.features}")

    # Assuming the column containing sentences is 'text'
    # This is standard for many text datasets on Hugging Face.
    # dataset['text'] directly gives a list of all values in the 'text' column.
    sentences_list = dataset['text']

    # If you want to be absolutely sure it's a Python list (it usually is or acts like one):
    # sentences_list = list(dataset['text'])

    # 4. Verify and print some examples
    print(f"\nSuccessfully extracted {len(sentences_list)} sentences into a list.")

    if sentences_list:
        print("\nFirst 5 sentences:")
        for i in range(min(5, len(sentences_list))):
            print(f"{i+1}. {sentences_list[i]}")

        print("\nLast 5 sentences:")
        for i in range(max(0, len(sentences_list) - 5), len(sentences_list)):
            print(f"{len(sentences_list) - (len(sentences_list) - 1 - i)}. {sentences_list[i]}")
    else:
        print("The list of sentences is empty.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have an active internet connection and the `datasets` library is installed.")
    print("Dataset name might be incorrect or the dataset might require authentication or specific configurations.")

entries = []
for sentence in ProgressBar(sentences_list, desc="Processing Items"):
    prompt = f"""Given the following text chunk:
    "{sentence}"

    Generate a JSON object with the following keys and corresponding string values:
    - "id": A title to the sentence being processed
    - "highly_similar": A paraphrase of the original chunk, maintaining the core meaning but using different wording and sentence structure.
    - "related": A sentence or short paragraph that is on the same general topic as the original chunk but discusses a different aspect or a related concept. It should not be a direct paraphrase.
    - "dissimilar": A sentence or short paragraph on a completely unrelated topic.
    - "question_form": A question that encapsulates the main idea or asks about a key aspect of the original chunk.
    - "negation": A sentence that negates the main assertion or a key aspect of the original chunk, while still being topically relevant if possible (e.g., not "The sky is not blue" if the topic is computers).

    Ensure the output is ONLY a valid JSON object. Example:
    {{
    "id": "...",
    "highly_similar": "...",
    "related": "...",
    "dissimilar": "...",
    "question_form": "...",
    "negation": "..."
    }}

    JSON object:
    """
    try:
        output = lc.generate_code(prompt)
        entry = json.loads(output)
        entry["query"]=sentence
        entries.append(entry)
        with open("benchmark_db.json","w") as f:
            json.dump(entries, f, indent=4)
    except Exception as ex:
        trace_exception(ex)