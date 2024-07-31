import os
from typing import List, Optional

import open_clip
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchFeature

from .base_interface import LMMInterface
from lmm_icl_interface.lmm_processor import OpenFlamingoPromptProcessor


class OpenFlamingoInterface(LMMInterface):
    def __init__(
        self,
        lang_encoder_path,
        tokenizer_path,
        flamingo_checkpoint_dir,
        cross_attn_every_n_layers,
        hf_root,
        precision,
        device,
        prompt_manager,
        instruction,
        image_field,
        label_field,
        load_from_local=False,
        init_device="cpu",
    ) -> None:
        from open_flamingo.src.factory import _infer_decoder_layers_attr_name
        from open_flamingo.src.flamingo import Flamingo
        from open_flamingo.src.flamingo_lm import FlamingoLMMixin
        from open_flamingo.src.utils import extend_instance

        super().__init__(
            precision=precision,
            device=device,
            input_ids_field_name="lang_x",
            prompt_manager=prompt_manager,
            instruction=instruction,
            label_field=label_field,
            image_field=image_field,
        )
        hf_device_map = {"transformer": self.device}

        self.model, _, _ = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=str(lang_encoder_path),
            tokenizer_path=str(tokenizer_path),
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            use_local_files=load_from_local,
            init_device=init_device,
            model_data_type=self.data_type,
            hf_device_map=hf_device_map,
        )
        self.processor = OpenFlamingoPromptProcessor(
            tokenizer_name_or_path=tokenizer_path
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.image_processor = self.processor.image_processor
        if load_from_local:
            flamingo_checkpoint_dir = os.path.join(
                flamingo_checkpoint_dir, "checkpoint.pt"
            )
        else:
            hf_root = "openflamingo/" + hf_root
            flamingo_checkpoint_dir = hf_hub_download(
                hf_root, "checkpoint.pt", local_dir=flamingo_checkpoint_dir
            )

        self.model.load_state_dict(torch.load(flamingo_checkpoint_dir), strict=False)

        self.model.to(device=device, dtype=self.data_type, non_blocking=True)
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    init_device="cpu",
    model_data_type=torch.bfloat16,
    hf_device_map="auto",
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
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
    if "llama-7b" in lang_encoder_path:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            local_files_only=use_local_files,
            trust_remote_code=True,
        )
    else:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            local_files_only=use_local_files,
            trust_remote_code=True,
            init_device=init_device,
            torch_dtype=model_data_type,
            device_map=hf_device_map,
        )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied

    logger.info(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
