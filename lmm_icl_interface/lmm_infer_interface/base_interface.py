from io import BytesIO
from typing import List

import requests
import torch
from loguru import logger
from PIL import Image
from torch import nn
from ..utils import cast_type, get_autocast, is_url
from lmm_icl_interface.prompt_control import LMMPromptManager


class BaseInterface(nn.Module):
    def __init__(
        self,
        precision,
        input_ids_field_name: str,
        prompt_manager: LMMPromptManager,
        instruction: str,
        label_field: str,
    ) -> None:
        super().__init__()
        self.data_type = cast_type(precision)
        self.autocast_context = get_autocast(precision)
        self.input_ids_field_name = input_ids_field_name

        self.prompt_manager = prompt_manager
        self.instruction = instruction
        self.pad_token_id = None
        self.tokenizer = None
        self.label_field = label_field

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        else:
            logger.warning("the model has not device parameters")
            return None

    @torch.inference_mode()
    def get_cond_prob(
        self,
        model_input,
        mask_length=None,
    ):
        ce_loss = self.get_ppl(model_input, mask_length)
        return (-ce_loss).exp()

    @torch.inference_mode()
    def get_ppl(
        self,
        model_input,
        mask_length=None,
    ):
        if self.pad_token_id is None:
            logger.warning("the pad_token_id is None")
        with self.autocast_context:
            outputs = self.model(**model_input)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = model_input[self.input_ids_field_name][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=self.pad_token_id
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())

            if mask_length is not None:
                loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
                for i in range(len(loss_mask)):
                    for j in range(mask_length[i] - 1, len(loss_mask[i])):
                        loss_mask[i][j] = 1
                loss = loss * loss_mask
            lens = (model_input[self.input_ids_field_name] != self.pad_token_id).sum(-1)

            if mask_length is not None:
                lens -= torch.tensor(mask_length, device=lens.device)

            ce_loss = loss.sum(-1) / lens
        return ce_loss

    def transfer_icl_prompts(
        self, batch_data_sample_list, is_last_for_generation=True, query_label=None
    ):
        """
        transfer data sample list to text input.
        Note: Only support one image and one text pair.
        """
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.extend(
                    [
                        data_sample[self.image_field],
                        self.prompt_manager.gen_ice_text_with_label(
                            data_sample, add_sep_token=True
                        ),
                    ]
                )
            prompt.append(data_sample_list[-1][self.image_field])
            if is_last_for_generation:
                prompt.append(
                    self.prompt_manager.gen_query_text_without_label(
                        data_sample_list[-1]
                    )
                )
            else:
                prompt.append(
                    self.prompt_manager.gen_query_text_with_label(
                        data_sample_list[-1], label=query_label
                    )
                )

            prompts.append(prompt)
        return prompts

    def generate(self, *args, **kwargs):
        with self.autocast_context:
            return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with self.autocast_context:
            return self.model(*args, **kwargs)


class LMMInterface(BaseInterface):
    def __init__(
        self,
        precision,
        input_ids_field_name,
        prompt_manager: LMMPromptManager,
        instruction,
        label_field,
        image_field,
    ):
        super().__init__(
            precision=precision,
            input_ids_field_name=input_ids_field_name,
            prompt_manager=prompt_manager,
            instruction=instruction,
            label_field=label_field,
        )

        self.image_field = image_field

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
