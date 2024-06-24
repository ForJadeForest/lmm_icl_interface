import open_clip
import torch
from transformers import AutoTokenizer, BatchFeature

from .base_processor import LMMPromptProcessor


class OpenFlamingoPromptProcessor(LMMPromptProcessor):
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
        )
        # add Flamingo special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        _, _, image_processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        super().__init__(text_tokenizer, image_processor)
        self.set_input_ids_field("lang_x")

    def prepare_input(
        self,
        batch_prompts,
        padding="longest",
        truncation=None,
        max_length=None,
        add_eos_token: bool = False,
        debug=False,
        return_tensors="pt",
        add_special_tokens=True,
    ):
        if not any(isinstance(i, list) for i in batch_prompts):
            batch_prompts = [batch_prompts]
        image_token = "<image>"

        all_images = []
        all_raw_texts = []
        for sample in batch_prompts:

            image_objects = []
            full_text = ""
            for i, item in enumerate(sample):
                item_is_img = self.is_img(item)
                if item_is_img is None:
                    item = item.strip(" ")
                    full_text += item
                else:
                    full_text += image_token
                    image_objects.append(item_is_img)

            if add_eos_token:
                full_text += self.tokenizer.eos_token

            if debug is True:
                print(f"{full_text=}")

            image_objects = torch.stack(
                [self.image_processor(image) for image in image_objects], dim=0
            )
            all_raw_texts.append(full_text)
            all_images.append(image_objects)

        # max_num_images has to be at least 1 even when there are no images
        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        output_input_ids = []
        output_images = []
        output_attention_masks = []

        text_tensor_input = self.tokenizer(
            all_raw_texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )
        for text_tensor, images in zip(text_tensor_input["input_ids"], all_images):
            image_count = (text_tensor == self.image_token_id).sum()

            local_max_num_images = min(image_count, max_num_images)
            current_images = images[:local_max_num_images]

            if len(current_images) > 0:
                padded_image_tensor = torch.zeros(
                    max_num_images, *current_images.size()[1:]
                )
                padded_image_tensor[: current_images.size(0)] = current_images
            else:
                padded_image_tensor = torch.zeros(
                    max_num_images, *self.default_image_dims
                )

            output_images.append(padded_image_tensor)

        output_input_ids = text_tensor_input["input_ids"]
        output_images = torch.stack(output_images)
        output_attention_masks = text_tensor_input["attention_mask"]

        return BatchFeature(
            data={
                "lang_x": output_input_ids,
                "attention_mask": output_attention_masks,
                "vision_x": output_images.unsqueeze(dim=2),
            }
        )
