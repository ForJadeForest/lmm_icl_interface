from .base_processor import LMMPromptProcessor
from transformers import IdeficsProcessor


class IdeficsPromptProcessor(LMMPromptProcessor):
    def __init__(self, model_name_or_path):
        self.processor = IdeficsProcessor.from_pretrained(model_name_or_path)
        super().__init__(self.processor.tokenizer, self.processor.image_processor)

    def prepare_input(
        self,
        batch_prompts,
        padding="longest",
        truncation=None,
        max_length=None,
        transform=None,
        add_eos_token=False,
        add_end_of_utterance_token=None,
        debug=False,
        return_tensors="pt",
    ):
        return self.processor(
            batch_prompts,
            padding,
            truncation,
            max_length,
            transform,
            add_eos_token,
            add_end_of_utterance_token,
            debug,
            return_tensors,
        )
