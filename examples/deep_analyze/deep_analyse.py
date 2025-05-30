from lollms_client import LollmsClient
import pipmaster as pm
from ascii_colors import ASCIIColors
if not pm.is_installed("docling"):
    pm.install("docling")
from docling.document_converter import DocumentConverter

ASCIIColors.set_log_file("log.log")

lc = LollmsClient()
# Create prompts for each section
article_url = "https://arxiv.org/pdf/2109.09572"
converter = DocumentConverter()
result = converter.convert(article_url)
article_text = result.document.export_to_markdown()

ASCIIColors.info("Article loaded successfully")

# Use the sequential_summarize method from lollms
result = lc.deep_analyze(
                    "Explain what is the difference between HGG and QGG",
                    article_text,
                    ctx_size=128000,
                    chunk_size=1024,
                    bootstrap_chunk_size=512,
                    bootstrap_steps=1,
                    debug = True
                )

print(result)
