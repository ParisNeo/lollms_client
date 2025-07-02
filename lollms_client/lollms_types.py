from  enum import Enum
class MSG_TYPE(Enum):
    # Messaging
    MSG_TYPE_CHUNK                      = 0 # A chunk of a message (used for classical chat)
    MSG_TYPE_CONTENT                   = 1 # A full message (for some personality the answer is sent in bulk)
    MSG_TYPE_CONTENT_INVISIBLE_TO_AI   = 2 # A full message (for some personality the answer is sent in bulk)
    MSG_TYPE_CONTENT_INVISIBLE_TO_USER = 3 # A full message (for some personality the answer is sent in bulk)

    # Thoughts
    MSG_TYPE_THOUGHT_CHUNK             = 4 # A chunk of a thought content (used for classical chat)
    MSG_TYPE_THOUGHT_CONTENT           = 5 # A full thought content (for some personality the answer is sent in bulk)

    # Conditionning
    # Informations
    MSG_TYPE_EXCEPTION              = 6 # An exception occured
    MSG_TYPE_WARNING                = 7 # A warning occured
    MSG_TYPE_INFO                   = 8 # An information to be shown to user

    # Steps
    MSG_TYPE_STEP                   = 9 # An instant step (a step that doesn't need time to be executed)
    MSG_TYPE_STEP_START             = 10 # A step has started (the text contains an explanation of the step done by he personality)
    MSG_TYPE_STEP_PROGRESS          = 11 # The progress value (the text contains a percentage and can be parsed by the reception)
    MSG_TYPE_STEP_END               = 12# A step has been done (the text contains an explanation of the step done by he personality)

    #Extra
    MSG_TYPE_JSON_INFOS             = 13# A JSON output that is useful for summarizing the process of generation used by personalities like chain of thoughts and tree of thooughts
    MSG_TYPE_REF                    = 14# References (in form of  [text](path))
    MSG_TYPE_CODE                   = 15# A javascript code to execute
    MSG_TYPE_UI                     = 16# A vue.js component to show (we need to build some and parse the text to show it)

    #Commands
    MSG_TYPE_NEW_MESSAGE            = 17# A new message
    MSG_TYPE_FINISHED_MESSAGE       = 18# End of current message

    #Tool calling
    MSG_TYPE_TOOL_CALL              = 19# a tool call
    MSG_TYPE_TOOL_OUTPUT            = 20# the output of the tool

    MSG_TYPE_REASONING              = 21# the ai shows its reasoning
    MSG_TYPE_SCRATCHPAD             = 22# the ai shows its scratchpad


class SENDER_TYPES(Enum):
    SENDER_TYPES_USER               = 0 # Sent by user
    SENDER_TYPES_AI                 = 1 # Sent by ai
    SENDER_TYPES_SYSTEM             = 2 # Sent by athe system


class SUMMARY_MODE(Enum):
    SUMMARY_MODE_SEQUENCIAL        = 0
    SUMMARY_MODE_HIERARCHICAL      = 0

class ELF_COMPLETION_FORMAT(Enum):
    Instruct = 0
    Chat = 1
    @classmethod
    def from_string(cls, format_string: str) -> 'ELF_COMPLETION_FORMAT':
        format_mapping = {
            "Instruct": cls.Instruct,
            "Chat": cls.Chat,
        }

        try:
            return format_mapping[format_string.upper()]
        except KeyError:
            raise ValueError(f"Invalid format string: {format_string}. Must be one of {list(format_mapping.keys())}.")
    
    def __str__(self):
        return self.name    