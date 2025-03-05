from lollms_client import LollmsClient, ELF_GENERATION_FORMAT
import pipmaster as pm
from ascii_colors import ASCIIColors
if not pm.is_installed("docling"):
    pm.install("docling")
from docling.document_converter import DocumentConverter

ASCIIColors.set_log_file("log.log")

lc = LollmsClient(
    host_address="http://lollms:11434",
    model_name="phi4:latest",
    ctx_size=32800,
    default_generation_mode=ELF_GENERATION_FORMAT.OLLAMA
)
# Create prompts for each section
article_url = "https://arxiv.org/pdf/2109.09572"
converter = DocumentConverter()
result = converter.convert(article_url)
article_text = result.document.export_to_markdown()

ASCIIColors.info("Article loaded successfully")

# Use the sequential_summarize method from lollms
summary = lc.sequential_summarize(
                    article_text,
                    """
Extract the following information if present in the chunk:

1. **Title**: 
   - Found in text chunk number 1 at the beginning. It should be followed by # or ##
   - Copy exactly as presented; do not interpret.
   - Never alter this if already in the memory. This is important

2. **Authors**: 
   - Listed in text chunk number 1 at the beginning.
   - If you fail to find the authors keep this empty.
   - Copy exactly as presented; do not interpret.
   - Never alter this if already in the memory. This is important

3. **Summary**: 
   - Provide a concise but detailed summary of the article by adding ned information from the text chunk to the memory content.

4. **Results**: 
   - Extract quantified results if available.

Ensure that any information already in memory is retained unless explicitly updated by the current chunk.
""",
                    "markdown",
                    """Write a final markdown with these sections:
## Title
## Authors
## Summary
## Results
                    """,
                    ctx_size=128000,
                    chunk_size=4096,
                    bootstrap_chunk_size=1024,
                    bootstrap_steps=1,
                    debug = True
                )

print(summary)
