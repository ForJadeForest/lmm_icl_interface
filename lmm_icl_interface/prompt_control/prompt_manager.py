from .prompt_template import PromptTemplate


class PromptManager:
    def __init__(
        self,
        prompt_template,
        column_token_map,
        label_field,
    ):
        if not isinstance(prompt_template, str):
            prompt_template = dict(prompt_template)
        self.pt = PromptTemplate(
            prompt_template,
            column_token_map=dict(column_token_map),
        )
        self.label_field = label_field

    def gen_text_with_label(self, item, label=None):
        if label is None and isinstance(self.pt.template, dict):
            label = item[self.label_field]

        return self.pt.generate_ice_item(item, label)

    def gen_text_without_label(self, item):
        return self.pt.generate_item(item, output_field=self.label_field)


class LMMPromptManager(PromptManager):
    def __init__(
        self,
        prompt_template,
        column_token_map,
        label_field,
        image_prompt,
    ):
        super().__init__(
            prompt_template,
            column_token_map,
            label_field,
        )
        self.image_prompt = image_prompt

    def add_image_token(self, text):
        return self.image_prompt + text

    def gen_text_with_label(self, item, label=None, add_image_token=False):
        prompt = super().gen_text_with_label(item, label)
        if add_image_token:
            return self.add_image_token(prompt)
        return prompt

    def gen_text_without_label(self, item, add_image_token=False):
        prompt = super().gen_text_without_label(item)
        if add_image_token:
            return self.add_image_token(prompt)
        return prompt
