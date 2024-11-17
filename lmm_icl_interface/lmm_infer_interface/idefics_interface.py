from typing import Optional

import torch
from loguru import logger
from transformers import BatchFeature, IdeficsForVisionText2Text, IdeficsProcessor

from .base_interface import LMMInterface
from lmm_icl_interface.lmm_processor import IdeficsPromptProcessor


class IdeficsInterface(LMMInterface):
    def __init__(
        self,
        model_name_or_path,
        precision,
        model_device,
        prompt_manager,
        instruction,
        image_field,
        label_field,
    ):
        super().__init__(
            precision=precision,
            input_ids_field_name="input_ids",
            prompt_manager=prompt_manager,
            instruction=instruction,
            label_field=label_field,
            image_field=image_field,
        )
        self.processor = IdeficsPromptProcessor(model_name_or_path)
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_name_or_path,
            torch_dtype=self.data_type,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.image_processor = self.processor.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id

        self.fake_token = "<fake_token_around_image>"
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
