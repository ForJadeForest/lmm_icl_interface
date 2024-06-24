from contextlib import suppress
from io import BytesIO
from typing import List
from urllib.parse import urlparse

import requests
import torch
from PIL import Image


def cast_type(precision):
    precision_list = ["fp16", "bf16", "fp32"]
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "fp32":
        return torch.float32
    else:
        raise ValueError(
            f"the precision should in {precision_list}, but got {precision}"
        )


def get_autocast(precision):
    if precision == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    elif precision == "bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def is_url(string):
    """Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url"""
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


def is_img(obj):
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
