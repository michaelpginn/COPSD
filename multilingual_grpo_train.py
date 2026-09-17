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

import re
from dataclasses import dataclass, field
from pathlib import Path

import wandb
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


LANGUAGE_PROMPTS = {
    # ============================================================
    # Existing PolyMath / MGSM-style languages
    # ============================================================
    "EN": {
        "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
        "think_prefix": "By request, I will start thinking in English.",
    },
    "ENG": {
        "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
        "think_prefix": "By request, I will start thinking in English.",
    },
    "DE": {
        "instruction": "Bitte denke Schritt für Schritt und setze deine endgültige Antwort in \\boxed{}.",
        "think_prefix": "Auf Anfrage werde ich anfangen, in Deutsch zu denken.",
    },
    "ES": {
        "instruction": "Por favor, razona paso a paso y coloca tu respuesta final dentro de \\boxed{}.",
        "think_prefix": "A petición, empezaré a pensar en español.",
    },
    "FR": {
        "instruction": "Veuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{}.",
        "think_prefix": "Sur demande, je commencerai à penser en français.",
    },
    "FRA": {
        "instruction": "Veuillez raisonner étape par étape et mettre votre réponse finale dans \\boxed{}.",
        "think_prefix": "Sur demande, je commencerai à penser en français.",
    },
    "JA": {
        "instruction": "段階的に考え、最終的な答えを \\boxed{} の中に入れてください。",
        "think_prefix": "要望があれば、日本語で考え始めます。",
    },
    "ZH": {
        "instruction": "请一步一步推理，并将最终答案放在 \\boxed{} 中。",
        "think_prefix": "应要求，我将开始用中文思考。",
    },
    "RU": {
        "instruction": "Пожалуйста, рассуждайте шаг за шагом и поместите окончательный ответ в \\boxed{}.",
        "think_prefix": "По запросу я начну думать на русском.",
    },
    "SW": {
        "instruction": "Tafadhali fikiri hatua kwa hatua, na uweke jibu lako la mwisho ndani ya \\boxed{}.",
        "think_prefix": "Kwa ombi, nitaanza kufikiria kwa Kiswahili.",
    },
    "BN": {
        "instruction": "দয়া করে ধাপে ধাপে ভাবুন এবং আপনার চূড়ান্ত উত্তর \\boxed{} এর মধ্যে দিন।",
        "think_prefix": "অনুরোধ করলে, আমি বাংলায় চিন্তা করা শুরু করব।",
    },
    "TE": {
        "instruction": "దయచేసి దశలవారీగా ఆలోచించి, మీ చివరి సమాధానాన్ని \\boxed{} లో పెట్టండి.",
        "think_prefix": "అభ్యర్థన మేరకు, నేను తెలుగులో ఆలోచించడం ప్రారంభిస్తాను.",
    },
    "TH": {
        "instruction": "โปรดให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายของคุณไว้ใน \\boxed{}",
        "think_prefix": "ตามคำขอ ฉันจะเริ่มคิดเป็นภาษาไทย",
    },
    # ============================================================
    # AfriMGSM African languages, ISO 639-3 uppercase keys
    # ============================================================
    "EWE": {
        "instruction": "Mesrɛ wo, bu akɔntaabu no afã afã, eye nàtsɔ wo ŋuɖoɖo mamlɛtɔ ade \\boxed{} me.",
        "think_prefix": "Le biabia me la, magɔme le susu wɔwɔ me le Eʋegbe me.",
    },
    "HAU": {
        "instruction": "Da fatan ka yi tunani mataki-mataki, kuma ka sanya amsarka ta ƙarshe a cikin \\boxed{}.",
        "think_prefix": "Bisa buƙata, zan fara yin tunani da Hausa.",
    },
    "IBO": {
        "instruction": "Biko tụlee ya nzọụkwụ site na nzọụkwụ, ma tinye azịza ikpeazụ gị n'ime \\boxed{}.",
        "think_prefix": "Dị ka arịrịọ si dị, aga m amalite iche echiche n'asụsụ Igbo.",
    },
    "TWI": {
        "instruction": "Yɛsrɛ wo, dwene ho anammɔn biara mu, na fa wo mmuae a etwa to no hyɛ \\boxed{} mu.",
        "think_prefix": "Sɛnea wɔabisa no, mɛfi ase adwene wɔ Twi mu.",
    },
    "VAI": {
        "instruction": "ꕉ ꕘꕌꘋꕡ, ꔤ ꕞꕌ ꔳꘋ ꔳꘋ ꗏ ꖴꘋꗒ, ꔤ ꕒꕌꘋ ꔞꘋꗣ ꕉ ꗓ ꕉ ꕉꕌꘋꔕ ꗏ \\boxed{}.",
        "think_prefix": "ꗋꖺ ꕉ ꖏꕎꔀ ꗏ, ꔤ ꕘꕌ ꖴꘋꗒ ꕉ ꗓ ꔞꔀ ꗏ.",
    },
    "WOL": {
        "instruction": "Ba beneen yoon, xalaatal ci ndànk-ndànk, te defal sa tontu mu mujj mi ci \\boxed{}.",
        "think_prefix": "Ci laaj bi, dinaa tàmbali xalaat ci Wolof.",
    },
    "YOR": {
        "instruction": "Jọ̀wọ́ ronú ní ìgbésẹ̀-ní-ìgbésẹ̀, kí o sì fi ìdáhùn ìkẹyìn rẹ sínú \\boxed{}.",
        "think_prefix": "Gẹ́gẹ́ bí a ti béèrè, màá bẹ̀rẹ̀ sí í ronú ní èdè Yorùbá.",
    },
    "AMH": {
        "instruction": "እባክዎ ደረጃ በደረጃ ያስቡ፣ የመጨረሻ መልስዎንም በ \\boxed{} ውስጥ ያስቀምጡ።",
        "think_prefix": "በጥያቄው መሠረት፣ በአማርኛ ማሰብ እጀምራለሁ።",
    },
    "KIN": {
        "instruction": "Nyamuneka tekereza intambwe ku yindi, kandi ushyire igisubizo cya nyuma muri \\boxed{}.",
        "think_prefix": "Nk'uko byasabwe, ngiye gutangira gutekereza mu Kinyarwanda.",
    },
    "LUG": {
        "instruction": "Nsaba lowooza mutendera ku mutendera, era oteeke eky'okuddamu ekisembayo mu \\boxed{}.",
        "think_prefix": "Nga bwe kisabiddwa, nja kutandika okulowooza mu Luganda.",
    },
    "SWA": {
        "instruction": "Tafadhali fikiri hatua kwa hatua, na uweke jibu lako la mwisho ndani ya \\boxed{}.",
        "think_prefix": "Kwa ombi, nitaanza kufikiria kwa Kiswahili.",
    },
    "ORM": {
        "instruction": "Maaloo tartiiba tartiibaan yaadi, deebii kee isa dhumaa \\boxed{} keessa kaa'i.",
        "think_prefix": "Gaaffii kanaan, Afaan Oromootiin yaaduu nan jalqaba.",
    },
    "SNA": {
        "instruction": "Ndapota funga nhanho nenhanho, uye isa mhinduro yako yekupedzisira mukati me \\boxed{}.",
        "think_prefix": "Sekukumbirwa, ndichatanga kufunga nechiShona.",
    },
    "XHO": {
        "instruction": "Nceda ucinge inyathelo ngenyathelo, uze ufake impendulo yakho yokugqibela ngaphakathi kwe \\boxed{}.",
        "think_prefix": "Ngokwesicelo, ndiza kuqalisa ukucinga ngesiXhosa.",
    },
    "ZUL": {
        "instruction": "Sicela ucabange isinyathelo ngesinyathelo, bese ufaka impendulo yakho yokugcina ngaphakathi kwe \\boxed{}.",
        "think_prefix": "Ngokwesicelo, ngizoqala ukucabanga ngesiZulu.",
    },
    "SOT": {
        "instruction": "Ka kopo nahana mohato ka mohato, 'me u kenye karabo ea hao ea ho qetela ka hare ho \\boxed{}.",
        "think_prefix": "Ho latela kopo, ke tla qala ho nahana ka Sesotho.",
    },
    "LIN": {
        "instruction": "Nabondeli yo, kanisá litambe na litambe, mpe tyá eyano na yo ya nsuka na kati ya \\boxed{}.",
        "think_prefix": "Na kolanda bosengi, nakobanda kokanisa na Lingála.",
    },
}


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with GRPO-specific options."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name) to log runs under."},
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "WandB project name to log runs under."},
    )
    translated_data_path: str = field(
        default="",
        metadata={
            "help": "Path to per-language translated JSON. If empty, use "
            "./translated_opsd/translated_full_<lang>.json"
        },
    )
    translated_data_dir: str = field(
        default="./translated_opsd",
        metadata={"help": "Directory containing translated_full_<lang>.json files."},
    )
    train_language: str = field(
        default="ES",
        metadata={
            "help": "Target language for GRPO training, e.g. ES, DE, FR, JA, ZH."
        },
    )
    require_translation_ok: bool = field(
        default=True,
        metadata={
            "help": "If True, require problem_<lang>_ok == True when that field exists."
        },
    )
    use_think_hack: bool = field(
        default=True,
        metadata={
            "help": "Whether to manually append Qwen3 <think> prefix with language-specific hacking."
        },
    )


def resolve_translated_data_path(script_args):
    if script_args.translated_data_path:
        return script_args.translated_data_path
    lang = script_args.train_language.lower()
    return str(Path(script_args.translated_data_dir) / f"translated_full_{lang}.json")


def extract_boxed_answer(text):
    """
    Extract the answer from \\boxed{} format.
    For thinking models, only searches after </think> to avoid picking up
    intermediate answers from the thinking block.
    Handles nested braces correctly.
    """
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    """Convert ratio notation a:b → \\frac{a}{b} so math_verify can parse it."""
    if answer is None:
        return None
    ratio_match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer
    )
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def reward_correctness(completions, Answer, **kwargs):
    rewards = []
    for completion, ground_truth in zip(completions, Answer):
        pred_answer = extract_boxed_answer(completion)
        reward = 0.0

        gold_parsed = parse(str(ground_truth))
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass

        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", str(ground_truth) or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)

    return rewards


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


def build_qwen3_prompt(tokenizer, question, language_code, use_think_hack=True):
    lang = language_code.upper()
    if lang not in LANGUAGE_PROMPTS:
        raise ValueError(f"Unsupported language code: {lang}")

    user_message = f"{question}\n\n{LANGUAGE_PROMPTS[lang]['instruction']}"
    messages = [{"role": "user", "content": user_message}]

    base_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if use_think_hack:
        return (
            base_prompt
            + "<|im_start|>assistant\n<think>\n"
            + LANGUAGE_PROMPTS[lang]["think_prefix"]
        )
    else:
        return base_prompt + "<|im_start|>assistant\n"


def make_format_prompt(tokenizer, train_language, use_think_hack=True):
    """
    Returns a formatting function for one specific target language.
    Expects each example to contain problem_<lang> and Answer.
    """
    lang = train_language.upper()
    lang_key = lang.lower()
    q_key = f"problem_{lang_key}"

    def format_prompt(example):
        question = example[q_key]
        prompt = build_qwen3_prompt(
            tokenizer=tokenizer,
            question=question,
            language_code=lang,
            use_think_hack=use_think_hack,
        )
        return {
            "prompt": prompt,
            "Answer": str(example["Answer"]),
        }

    return format_prompt


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig))
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
        full_wandb_run_name = f"{script_args.run_config}_{target_lang}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(
                Path(training_args.output_dir)
                / f"{script_args.run_config}_{target_lang.lower()}"
            )
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_name = (
            f"GRPO_{model_name}_{target_lang.lower()}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )
        training_args.output_dir = str(
            Path(training_args.output_dir) / target_lang.lower()
        )

    print(f"\n{'=' * 80}")
    print("RUN CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Target Language: {target_lang}")
    print(f"Translated Data Path: {translated_data_path}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Prompt Length: {training_args.max_prompt_length}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"{'=' * 80}\n")

    ################
    # WandB Initialization
    ################
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "translated_data_path": translated_data_path,
                "train_language": target_lang,
                "require_translation_ok": script_args.require_translation_ok,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_prompt_length": training_args.max_prompt_length,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "use_think_hack": script_args.use_think_hack,
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
            f"Check your translated JSON and problem_{target_lang.lower()}."
        )

    format_prompt = make_format_prompt(
        tokenizer=tokenizer,
        train_language=target_lang,
        use_think_hack=script_args.use_think_hack,
    )
    train_dataset = train_dataset.map(
        format_prompt, remove_columns=train_dataset.column_names
    )

    split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = sorted(
            [
                d
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-")
            ],
            key=lambda x: int(x.split("-")[-1]),
        )
        if checkpoints:
            resume_from_checkpoint = os.path.join(
                training_args.output_dir, checkpoints[-1]
            )
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
