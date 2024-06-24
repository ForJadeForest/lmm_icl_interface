from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image


def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


class LMMPromptProcessor:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.input_ids_field = "input_ids"

    def set_input_ids_field(self, name: str):
        if isinstance(name, str):
            self.input_ids_field = name

    def prepare_input(self, *args, **kwargs):
        pass

    def is_img(self, obj):
        if isinstance(obj, Image.Image):
            return obj
        elif isinstance(obj, str):
            if is_url(obj):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                        " Safari/537.36"
                    )
                }
                response = requests.get(obj, stream=True, headers=headers)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                try:
                    return Image.open(obj)
                except:
                    return None

    def get_input_token_num(self, input_tokens: str) -> int:
        return len(self.tokenizer(input_tokens, add_special_tokens=False)["input_ids"])
