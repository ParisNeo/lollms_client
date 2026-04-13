from enum import Enum

class MSG_TYPE(Enum):
    # Messaging
    MSG_TYPE_CHUNK                      = 0  # A chunk of a message (used for classical chat)
    MSG_TYPE_CONTENT                    = 1  # A full message (for some personality the answer is sent in bulk)
    MSG_TYPE_CONTENT_INVISIBLE_TO_AI    = 2  # A full message (for some personality the answer is sent in bulk)
    MSG_TYPE_CONTENT_INVISIBLE_TO_USER  = 3  # A full message (for some personality the answer is sent in bulk)

    # Thoughts
    MSG_TYPE_THOUGHT_CHUNK              = 4  # A chunk of a thought content (used for classical chat)
    MSG_TYPE_THOUGHT_CONTENT            = 5  # A full thought content (for some personality the answer is sent in bulk)

    # Informations
    MSG_TYPE_EXCEPTION                  = 6  # An exception occured
    MSG_TYPE_WARNING                    = 7  # A warning occured
    MSG_TYPE_INFO                       = 8  # An information to be shown to user

    # Steps
    MSG_TYPE_STEP                       = 9  # An instant step (a step that doesn't need time to be executed)
    MSG_TYPE_STEP_START                 = 10 # A step has started
    MSG_TYPE_STEP_PROGRESS              = 11 # The progress value (text contains a percentage)
    MSG_TYPE_STEP_END                   = 12 # A step has been done

    # Extra
    MSG_TYPE_JSON_INFOS                 = 13 # A JSON output useful for summarizing the process
    MSG_TYPE_REF                        = 14 # References (in form of [text](path))
    MSG_TYPE_CODE                       = 15 # A javascript code to execute
    MSG_TYPE_UI                         = 16 # A vue.js component to show

    # Commands
    MSG_TYPE_NEW_MESSAGE                = 17 # A new message
    MSG_TYPE_FINISHED_MESSAGE           = 18 # End of current message

    # Tool calling
    MSG_TYPE_TOOL_CALL                  = 19 # a tool call
    MSG_TYPE_TOOL_OUTPUT                = 20 # the output of the tool

    MSG_TYPE_REASONING                  = 21 # the ai shows its reasoning
    MSG_TYPE_SCRATCHPAD                 = 22 # the ai shows its scratchpad
    MSG_TYPE_OBSERVATION                = 23 # the ai shows its reasoning

    MSG_TYPE_ERROR                      = 24 # a severe error happened
    MSG_TYPE_GENERATING_TITLE_START     = 25 # title generation started
    MSG_TYPE_GENERATING_TITLE_END       = 26 # title generation done

    MSG_TYPE_SOURCES_LIST               = 27 # List of sources provided

    MSG_TYPE_INIT_PROGRESS              = 28 # Initialization progress
    MSG_TYPE_ARTEFACTS_STATE_CHANGED    = 29 # Artefact was created, updated, or reverted

    MSG_TYPE_TOOLS_LIST                 = 30 # Structured list of all available tools
    MSG_TYPE_CONTEXT_COMPRESSION        = 31 # Context was compressed

    # ── Swarm events (added by lollms_swarm.py at import time) ───────────────
    # Values 32–37 are reserved for swarm events and registered dynamically.
    # Documented here for reference only — do not hardcode these values.
    #
    # MSG_TYPE_SWARM_AGENT_START        = 32
    # MSG_TYPE_SWARM_AGENT_END          = 33
    # MSG_TYPE_SWARM_ROUND_START        = 34
    # MSG_TYPE_SWARM_ROUND_END          = 35
    # MSG_TYPE_SWARM_HLF                = 36
    # MSG_TYPE_SWARM_CONSENSUS          = 37

    # ── Secondary content streams ─────────────────────────────────────────────
    # These events carry the raw content being built inside XML tags
    # (<artefact>, <note>, <skill>) on a SEPARATE stream
    # from the main chat bubble (MSG_TYPE_CHUNK).
    #
    # Contract
    # --------
    # For each tag the LLM opens, the framework fires:
    #   1. An "open" announcement on MSG_TYPE_CHUNK with meta={type:"artefact_update"|
    #      "note_start"|"skill_start"|"inline_widget_start"} — no content payload.
    #   2. Zero or more *_CHUNK events carrying raw content fragments as they stream.
    #   3. One *_DONE event carrying the complete raw content + full attribute meta.
    #
    # The *_DONE event fires BEFORE _post_process_llm_response runs, giving the app
    # the opportunity to preview or cache the content independently. Post-processing
    # still runs afterward and fires MSG_TYPE_ARTEFACTS_STATE_CHANGED as usual.
    #
    # meta dict shapes
    # ----------------
    # MSG_TYPE_ARTEFACT_CHUNK:
    #   {"title": str, "chunk": str, "art_type": str, "language": str|None}
    #
    # MSG_TYPE_ARTEFACT_DONE:
    #   {"title": str, "content": str, "art_type": str, "language": str|None,
    #    "is_patch": bool,   ← True when content contains <<<<<<< SEARCH markers
    #    "attrs": dict}      ← all other XML attributes from the opening tag
    #
    # MSG_TYPE_NOTE_CHUNK:
    #   {"title": str, "chunk": str}
    #
    # MSG_TYPE_NOTE_DONE:
    #   {"title": str, "content": str}
    #
    # MSG_TYPE_SKILL_CHUNK:
    #   {"title": str, "chunk": str, "category": str, "description": str}
    #
    # MSG_TYPE_SKILL_DONE:
    #   {"title": str, "content": str, "category": str, "description": str}
    #
    # MSG_TYPE_WIDGET_CHUNK:
    #   {"title": str, "chunk": str, "widget_type": str}
    #
    # MSG_TYPE_WIDGET_DONE:
    #   {"title": str, "content": str, "widget_type": str}
    #   NOTE: content here is the raw HTML/CSS/JS source validated on arrival.
    #   The app should NOT attempt to render/mount it inline — wait for the
    #   lollms_widget anchor in the final message and use metadata["inline_widgets"].

    MSG_TYPE_ARTEFACT_CHUNK             = 38 # streaming content chunk of an artefact being built
    MSG_TYPE_ARTEFACT_DONE              = 39 # complete raw artefact content ready (pre-post-processing)
    MSG_TYPE_NOTE_CHUNK                 = 40 # streaming content chunk of a note being built
    MSG_TYPE_NOTE_DONE                  = 41 # complete raw note content ready
    MSG_TYPE_SKILL_CHUNK                = 42 # streaming content chunk of a skill being built
    MSG_TYPE_SKILL_DONE                 = 43 # complete raw skill content ready
    MSG_TYPE_WIDGET_CHUNK               = 44 # streaming content chunk of a widget being built
    MSG_TYPE_WIDGET_DONE                = 45 # complete raw widget source (validated HTML/CSS/JS only)

    MSG_TYPE_FORM_READY                 = 46 # complete parsed form descriptor ready for rendering
    MSG_TYPE_FORM_SUBMITTED             = 47 # user answers injected back into generation context
    # ── Form / Frame system ───────────────────────────────────────────────────
    # lollms_form lets the LLM ask the user structured questions that render as
    # an interactive form in the UI.  When submitted, the answers are sent back
    # to the LLM as a tool_result so generation can continue.
    #
    # Lifecycle
    # ---------
    #   1. MSG_TYPE_FORM_READY fires when </lollms_form> is fully buffered.
    #      meta["form"] contains the complete parsed form descriptor dict.
    #      Generation is PAUSED — the LLM cannot continue until answers arrive.
    #
    #   2. The application renders the form, collects user answers, and calls:
    #        discussion.submit_form_response(form_id, answers_dict)
    #      This resumes generation with the answers injected as a user/system message.
    #
    #   3. MSG_TYPE_FORM_SUBMITTED fires when the answers are injected, confirming
    #      the round-trip is complete.
    #
    # Form descriptor schema (meta["form"])
    # --------------------------------------
    # {
    #   "id":          str    – stable UUID for this form instance
    #   "title":       str    – displayed as the form heading
    #   "description": str    – optional subtitle / instructions for the user
    #   "submit_label":str    – label for the submit button (default "Submit")
    #   "fields":      list[FieldDescriptor]
    # }
    #
    # FieldDescriptor schema
    # ----------------------
    # {
    #   "name":        str    – machine key returned in the answers dict
    #   "label":       str    – human-readable label shown next to the field
    #   "type":        str    – see Field Types below
    #   "required":    bool   – default True
    #   "default":     any    – pre-filled value (optional)
    #   "placeholder": str    – hint text (text / textarea / number fields)
    #   "options":     list   – for select / radio / checkbox_group / rating
    #                           each item: str  OR  {"value": any, "label": str}
    #   "min":         number – for number / range / rating
    #   "max":         number – for number / range / rating
    #   "step":        number – for number / range
    #   "rows":        int    – for textarea (default 4)
    #   "accept":      str    – for file ("image/*", ".pdf", etc.)
    #   "multiple":    bool   – for file / select (allow multiple selections)
    #   "hint":        str    – small explanatory text shown below the field
    # }
    #
    # Field Types
    # -----------
    #   text          – single-line text input
    #   textarea      – multi-line text (rows controlled by "rows")
    #   number        – numeric input (int or float)
    #   range         – slider (requires min/max, optional step)
    #   select        – dropdown (single selection unless multiple=True)
    #   radio         – radio button group (single selection)
    #   checkbox       – single boolean checkbox
    #   checkbox_group – multiple checkboxes from an options list
    #   date          – date picker
    #   time          – time picker
    #   color         – color picker (returns #RRGGBB)
    #   rating        – star rating (min defaults to 1, max defaults to 5)
    #   file          – file upload (returns base64 data-URI or filename)
    #   hidden        – not shown to user; value comes from "default"
    #   code          – multi-line code editor with optional syntax highlighting
    #                   (language set via "language" field)
    #   section       – visual divider / heading with no user input;
    #                   "label" becomes a sub-heading


class SENDER_TYPES(Enum):
    SENDER_TYPES_USER               = 0  # Sent by user
    SENDER_TYPES_AI                 = 1  # Sent by ai
    SENDER_TYPES_SYSTEM             = 2  # Sent by the system


class SUMMARY_MODE(Enum):
    SUMMARY_MODE_SEQUENCIAL         = 0
    SUMMARY_MODE_HIERARCHICAL       = 0


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
            raise ValueError(
                f"Invalid format string: {format_string}. "
                f"Must be one of {list(format_mapping.keys())}."
            )

    def __str__(self):
        return self.name