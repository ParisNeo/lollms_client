from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client import LollmsPersonality
from lollms_client import MSG_TYPE, ELF_GENERATION_FORMAT
from ascii_colors import ASCIIColors
# Callback send
def cb(chunk, type: MSG_TYPE):
    print(chunk,end="", flush=True)

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600",default_generation_mode=ELF_GENERATION_FORMAT.OPENAI)
# Bu_ild inline personality
p = LollmsPersonality(
                        lc, 
                        "./personality/test/work_dir", 
                        "./personality/test/config_dir", 
                        cb, 
                        None,
                        author="ParisNeo",
                        name="test_persona",
                        user_name="user",
                        category="generic",
                        category_desc="generic stuff",
                        language="English",
                        personality_conditioning="!@>system: Act as a helper to the user.",
                        welcome_message="Hi, I'm your helper. Let me help you",

                        )
d = LollmsDiscussion(lc)
prompt=""
ASCIIColors.green("To quit press q")
ASCIIColors.yellow(p.welcome_message)
while prompt!="q":
    prompt = input("user:")
    if prompt=="q":
        break
    p.generate(d,prompt,stream=True)
    print("")