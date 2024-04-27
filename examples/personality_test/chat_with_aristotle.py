from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client import LollmsPersonality
from lollms_client import MSG_TYPE, ELF_GENERATION_FORMAT
from ascii_colors import ASCIIColors
# Callback send
def cb(chunk, type: MSG_TYPE):
    print(chunk,end="", flush=True)

# Initialize the LollmsClient instance
lc = LollmsClient("http://localhost:9600",default_generation_mode=ELF_GENERATION_FORMAT.LOLLMS)
# Bu_ild inline personality
aristotle_personality  = LollmsPersonality(
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
                        personality_conditioning="!@>system: Act as the philosopher Aristotle, sharing wisdom and engaging in logical discussions.",
                        welcome_message="Greetings, I am Aristotle, your guide in the pursuit of knowledge. How may I assist you in your philosophical inquiries?",
                    )
# Create a Discussion instance for Aristotle
aristotle_discussion = LollmsDiscussion(lc)

# Initialize user prompt
prompt = ""

# Print welcome message in yellow
ASCIIColors.yellow(aristotle_personality.welcome_message)

# Interaction loop
while prompt.lower() != "q":
    prompt = input("student: ")
    if prompt.lower() == "q":
        break
    aristotle_personality.generate(aristotle_discussion, prompt, stream=True)
    print("")