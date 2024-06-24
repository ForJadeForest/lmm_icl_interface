from .base_processor import LMMPromptProcessor
from transformers import AutoProcessor, Idefics2Processor
from ..utils import is_img


class Idefics2PromptProcessor(LMMPromptProcessor):
    def __init__(self, model_name_or_path, do_image_splitting=False):
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, do_image_splitting=do_image_splitting
        )
        super().__init__(self.processor.tokenizer, self.processor.image_processor)

    def prepare_input(
        self,
        batch_prompts,
        padding="longest",
        truncation=None,
        max_length=None,
        return_tensors="pt",
        add_special_tokens=True,
        add_eos_token=False,
    ):
        batch_text_inputs = []
        batch_image_inputs = []
        for prompts in batch_prompts:
            image_inputs = []
            text_inputs = ""
            for item in prompts:
                if is_img(item):
                    image_inputs.append(item)
                    text_inputs += "<image>"
                else:
                    text_inputs += item
            if add_eos_token:
                text_inputs += "</s>"
            batch_text_inputs.append(text_inputs)
            batch_image_inputs.append(image_inputs)

        return self.processor(
            text=batch_text_inputs,
            images=batch_image_inputs,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
