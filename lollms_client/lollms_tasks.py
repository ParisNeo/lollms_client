from lollms_client.lollms_core import LollmsClient


class TasksLibrary:
    def __init__(self, lollms:LollmsClient) -> None:
        self.lollms = lollms

    def translate_text_chunk(self, text_chunk, output_language:str="french", host_address:str=None, model_name: str = None, temperature=0.1, max_generation_size=3000):
        translated = self.lollms.generate_text(
                                "\n".join([
                                    f"!@>system:",
                                    f"Translate the following text to {output_language}.",
                                    "Be faithful to the original text and do not add or remove any information.",
                                    "Respond only with the translated text.",
                                    "Do not add comments or explanations.",
                                    f"!@>text to translate:",
                                    f"{text_chunk}",
                                    f"!@>translation:",
                                    ]),
                                    host_address,
                                    model_name,
                                    personality = -1,
                                    n_predict=max_generation_size,
                                    stream=False,
                                    temperature=temperature
                                    )
        return translated
