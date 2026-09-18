import re

import torch

from language_config import LANGUAGE_CONFIG


class MultilingualSelfDistillationDataCollator:
    """
    Qwen3-only COPSD data collator.

    Expected dataset fields:
      - problem                : English source question
      - solution               : English source solution
      - source                 : optional
      - Answer                 : optional
      - problem_bn / problem_de / ... / problem_zh
      - target_lang (preferred) or lang / language / lang_code

    Student:
      - sees only target-language question
      - gets target-language instruction
      - prompt is manually ended with:
            <|im_start|>assistant
            <think>
            [language-specific hack prefix]

    Teacher:
      - sees target-language question + English question + English solution
      - same manual think hack
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_enable_thinking=False,
        include_problem_en=True,
        include_reference_solution_en=True,
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_enable_thinking = student_enable_thinking
        self.include_problem_en = include_problem_en
        self.include_reference_solution_en = include_reference_solution_en

        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Include English problem: {self.include_problem_en}")
        print(
            f"[DataCollator] Include English reference solution: {self.include_reference_solution_en}"
        )

    def _normalize_lang(self, lang_code):
        if lang_code is None:
            return "EN"
        lang = str(lang_code).strip().upper()
        if lang not in LANGUAGE_CONFIG:
            print(
                f"[DataCollator] Warning: unsupported lang_code={lang}; falling back to EN."
            )
            return "EN"
        return lang

    def _get_lang(self, feature):
        for key in ["target_lang", "lang", "language", "lang_code"]:
            if key in feature and feature[key] is not None:
                return self._normalize_lang(feature[key])
        return "EN"

    def _get_problem_for_lang(self, feature, lang_code):
        if lang_code == "EN":
            return feature.get("problem_en", feature["problem"])

        lang_key = f"problem_{lang_code.lower()}"
        if feature.get(lang_key):
            return feature[lang_key]

        if feature.get("problem_en"):
            print(
                f"[DataCollator] Warning: missing {lang_key}; falling back to problem_en."
            )
            return feature["problem_en"]

        print(f"[DataCollator] Warning: missing {lang_key}; falling back to problem.")
        return feature["problem"]

    def _append_qwen3_assistant_prefix(self, base_prompt):
        return base_prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _append_qwen3_think_prefix(self, base_prompt, lang_code):
        """
        For Qwen3, manually append assistant generation prefix and think-hack.
        """
        think_prefix = LANGUAGE_CONFIG[lang_code]["think_prefix"].strip()
        return base_prompt + "<|im_start|>assistant\n<think>\n" + think_prefix

    def _tokenize_with_batch_max(self, texts):
        encoded_no_pad = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_len = max(lengths)

        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return encoded, lengths, max_len

    def _build_teacher_privileged_context(
        self,
        labels,
        problem_target,
        problem_en,
        solution_en,
    ):
        parts = [
            f"{labels['problem_target']}: {problem_target}",
        ]

        if self.include_problem_en:
            parts.append(f"{labels['problem_english']}: {problem_en}")

        if self.include_reference_solution_en:
            print(solution_en)
            answer_only = re.match(
                r".*\\\[\s+\\boxed{(.*)}\s+\\\].*", solution_en, flags=re.DOTALL
            ).group(1)
            # don't include boxed so there's no format clue
            parts.append(
                f"{labels['solution_english']}:\n{answer_only}"
                # f"{labels['ref_begin']}\n"
                # f"{solution_en}\n"
                # f"{labels['ref_end']}"
            )

        return "\n\n".join(parts)

    def _get_teacher_instruction(self, lang_cfg):
        # if self.include_reference_solution_en:
        #     return (
        #         f"{lang_cfg['transition_prompt']}\n"
        #         f"{lang_cfg['teacher_final_instruction']}"
        #     )

        return lang_cfg["teacher_final_instruction"]

    def __call__(self, features):
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []
        teacher_transition_texts = []
        lang_codes = []

        for feature in features:
            lang_code = self._get_lang(feature)
            lang_cfg = LANGUAGE_CONFIG[lang_code]
            labels = lang_cfg["labels"]

            lang_codes.append(lang_code)

            problem_en = feature.get("problem_en", feature["problem"])
            solution_en = feature["solution"]
            problem_target = self._get_problem_for_lang(feature, lang_code)

            teacher_context = self._build_teacher_privileged_context(
                labels=labels,
                problem_target=problem_target,
                problem_en=problem_en,
                solution_en=solution_en,
            )

            # -------------------------
            # Student prompt
            # -------------------------
            student_user_message = (
                f"{labels['problem_target']}: {problem_target}\n\n"
                f"{lang_cfg['student_instruction']}"
            )
            student_messages = [{"role": "user", "content": student_user_message}]

            student_base_prompt = self.tokenizer.apply_chat_template(
                student_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if self.student_enable_thinking:
                student_prompt = self._append_qwen3_think_prefix(
                    student_base_prompt, lang_code
                )
            else:
                student_prompt = self._append_qwen3_assistant_prefix(
                    student_base_prompt
                )

            student_prompts.append(student_prompt)
            # -------------------------
            # Teacher prompt(s)
            # -------------------------
            if self.reason_first:
                teacher_reasoning_user_message = (
                    f"{teacher_context}\n\n{lang_cfg['reason_first_prompt']}"
                )

                teacher_reasoning_messages = [
                    {"role": "user", "content": teacher_reasoning_user_message}
                ]
                teacher_reasoning_base_prompt = self.tokenizer.apply_chat_template(
                    teacher_reasoning_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                teacher_reasoning_prompt = self._append_qwen3_think_prefix(
                    teacher_reasoning_base_prompt, lang_code
                )
                teacher_reasoning_prompts.append(teacher_reasoning_prompt)

                teacher_transition_text = f"\n{self._get_teacher_instruction(lang_cfg)}"
                teacher_transition_texts.append(teacher_transition_text)

                teacher_prompts.append("")  # placeholder for compatibility
            else:
                teacher_user_message = (
                    f"{teacher_context}\n\n{self._get_teacher_instruction(lang_cfg)}"
                )

                teacher_messages = [{"role": "user", "content": teacher_user_message}]
                teacher_base_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                teacher_prompt = self._append_qwen3_think_prefix(
                    teacher_base_prompt, lang_code
                )
                teacher_prompts.append(teacher_prompt)

        # -------------------------
        # Student tokenization
        # -------------------------
        student_encoded, student_prompt_lengths, max_student_prompt_len = (
            self._tokenize_with_batch_max(student_prompts)
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(
                student_prompt_lengths, dtype=torch.long
            ),
            "lang_codes": lang_codes,
        }

        # -------------------------
        # Teacher tokenization
        # -------------------------
        if self.reason_first:
            reasoning_encoded, reasoning_prompt_lengths, max_reasoning_prompt_len = (
                self._tokenize_with_batch_max(teacher_reasoning_prompts)
            )

            transition_encoded, transition_lengths, max_transition_len = (
                self._tokenize_with_batch_max(teacher_transition_texts)
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded[
                        "attention_mask"
                    ],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_reasoning_prompt_lengths_per_example": torch.tensor(
                        reasoning_prompt_lengths, dtype=torch.long
                    ),
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                    "teacher_transition_attention_mask": transition_encoded[
                        "attention_mask"
                    ],
                    "teacher_transition_length": max_transition_len,
                    "teacher_transition_lengths_per_example": torch.tensor(
                        transition_lengths, dtype=torch.long
                    ),
                }
            )
        else:
            teacher_encoded, teacher_prompt_lengths, max_teacher_prompt_len = (
                self._tokenize_with_batch_max(teacher_prompts)
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(
                        teacher_prompt_lengths, dtype=torch.long
                    ),
                }
            )

        return result
