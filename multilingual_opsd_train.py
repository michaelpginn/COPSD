import os

os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

CACHE_ROOT = "/home/ec2-user/COPSD/.cache"

# ====== Hugging Face ======
os.environ["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
os.environ["HF_HUB_CACHE"] = f"{CACHE_ROOT}/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_ROOT}/huggingface/datasets"

# ====== vLLM======
os.environ["VLLM_CACHE_ROOT"] = f"{CACHE_ROOT}/vllm"

from dataclasses import dataclass, field
from pathlib import Path

import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from multilingual_opsd_trainer import OPSDTrainer

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments for multilingual COPSD."""

    translated_data_path: str = field(
        default="",
        metadata={
            "help": "Path to the per-language translated JSON file. "
            "If empty, defaults to ./translated_opsd/translated_full_<lang>.json"
        },
    )
    translated_data_dir: str = field(
        default="./translated_opsd",
        metadata={
            "help": "Directory containing per-language translated JSON files such as translated_full_de.json."
        },
    )
    train_language: str = field(
        default="DE",
        metadata={
            "help": "Target language to train on, e.g. DE, FR, JA, ZH, RU, SW, BN, TE, TH, ES, EN."
        },
    )
    require_translation_ok: bool = field(
        default=True,
        metadata={
            "help": "If True, filter out rows where problem_<lang>_ok is present and not True."
        },
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explicitly) "
            "about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. "
            "Both student and teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "whether allow the student model to generate thinking traces."
        },
    )

    include_problem_en: bool = field(
        default=True,
        metadata={
            "help": "Whether to include the English source problem in the teacher prompt. "
            "Default True matches the original COPSD setting."
        },
    )
    include_reference_solution_en: bool = field(
        default=True,
        metadata={
            "help": "Whether to include the English reference solution in the teacher prompt. "
            "Default True matches the original COPSD setting."
        },
    )


def resolve_translated_data_path(script_args) -> str:
    if script_args.translated_data_path:
        return script_args.translated_data_path

    lang = script_args.train_language.lower()
    return str(Path(script_args.translated_data_dir) / f"translated_full_{lang}.json")


def add_target_language(example, lang):
    example["target_lang"] = lang
    # English source question is stored in `problem` for your per-language JSONs.
    example["problem_en"] = example["problem"]
    return example


def has_target_translation(example, lang, require_ok=True):
    lang = lang.lower()
    q_key = f"problem_{lang}"
    ok_key = f"problem_{lang}_ok"

    if q_key not in example:
        return False
    if not isinstance(example[q_key], str) or not example[q_key].strip():
        return False

    if require_ok and ok_key in example:
        return example[ok_key] is True

    return True


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    script_args.train_language = script_args.train_language.upper()
    target_lang = script_args.train_language
    translated_data_path = resolve_translated_data_path(script_args)

    ################
    # WandB Run Name & Output Directory
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_processes
    )

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_{target_lang}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(
                Path(training_args.output_dir)
                / f"{script_args.run_config}_{target_lang.lower()}"
            )
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_config = (
            f"copsd_{model_name}_{target_lang.lower()}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

        training_args.output_dir = str(
            Path(training_args.output_dir) / target_lang.lower()
        )
        if not script_args.include_problem_en:
            full_wandb_run_config += "_no_en_problem"

        if not script_args.include_reference_solution_en:
            full_wandb_run_config += "_no_en_solution"

    print(f"\n{'=' * 80}")
    print("RUN CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Target Language: {target_lang}")
    print(f"Include English Problem: {script_args.include_problem_en}")
    print(
        f"Include English Reference Solution: {script_args.include_reference_solution_en}"
    )
    print(f"Translated Data Path: {translated_data_path}")
    print(f"{'=' * 80}\n")

    ################
    # WandB Initialization
    ################
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. "
            "As the fixed teacher is implemented by disabling LoRA adapters."
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "translated_data_path": translated_data_path,
                "train_language": target_lang,
                "require_translation_ok": script_args.require_translation_ok,
                "include_problem_en": script_args.include_problem_en,
                "include_reference_solution_en": script_args.include_reference_solution_en,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss
                if script_args.top_k_loss > 0
                else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay
                if script_args.use_ema_teacher
                else None,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'=' * 80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(
        f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}"
    )
    print(f"{'=' * 80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    training_args.presence_penalty = script_args.presence_penalty

    # IMPORTANT: skip SFTTrainer's default text-field preprocessing.
    # Our multilingual collator consumes raw rows directly.
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # IMPORTANT: keep all raw columns for the custom multilingual collator.
    # Without this, Trainer may remove columns like problem_ewe before batching.
    training_args.remove_unused_columns = False

    # if not hasattr(training_args, "dataset_text_field") or training_args.dataset_text_field is None:
    #     training_args.dataset_text_field = "problem"

    print(f"Loading translated JSON dataset from: {translated_data_path}")
    dataset = load_dataset("json", data_files=translated_data_path)
    train_dataset = dataset["train"]

    before_count = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda ex: has_target_translation(
            ex,
            lang=target_lang,
            require_ok=script_args.require_translation_ok,
        )
    )

    train_dataset = train_dataset.map(lambda ex: add_target_language(ex, target_lang))

    after_count = len(train_dataset)

    print(f"\n{'=' * 80}")
    print("DATASET SUMMARY")
    print(f"{'=' * 80}")
    print(f"Original examples: {before_count}")
    print(f"Usable examples for {target_lang}: {after_count}")
    print(f"Dropped examples: {before_count - after_count}")
    print(f"{'=' * 80}\n")

    if after_count == 0:
        raise ValueError(
            f"No usable examples found for target language {target_lang}. "
            f"Check your translated JSON and the problem_{target_lang.lower()} columns."
        )

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip
        if script_args.jsd_token_clip > 0
        else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_enable_thinking=script_args.student_enable_thinking,
        include_problem_en=script_args.include_problem_en,
        include_reference_solution_en=script_args.include_reference_solution_en,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(
            trainer, generation_config, num_prompts=8
        )
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
