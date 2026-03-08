from lollms_client import LollmsClient
from pathlib import Path
import pipmaster as pm
from ascii_colors import ASCIIColors

ASCIIColors.set_log_file("log.log")
def load_and_analyze_files():
    folder_path = Path('.')  # Change '.' to your desired directory
    allowed_extensions = {'.pdf', '.txt', '.md', '.docx', '.pptx', '.html'}
    
    matching_files = []
    for file in folder_path.rglob('*'):
        if file.suffix.lower() in allowed_extensions and file.is_file():
            matching_files.append(str(file.absolute()))
            
    # Now use these files with LollmsClient
    lc = LollmsClient()
    ASCIIColors.info(f"Loading {len(matching_files)} files for analysis")
    
    result = lc.deep_analyze(
        "Explain what is the difference between HGG and QGG",
        files=matching_files,
        ctx_size=128000,
        chunk_size=1024,
        bootstrap_chunk_size=512,
        bootstrap_steps=1,
        debug=True
    )
    
    print(result)

load_and_analyze_files()
