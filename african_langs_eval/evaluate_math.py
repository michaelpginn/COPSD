import os

os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

CACHE_ROOT = "/home/ec2-user/COPSD/.cache"

# ====== Hugging Face ======
os.environ["HF_HOME"] = f"{CACHE_ROOT}/huggingface"
os.environ["HF_HUB_CACHE"] = f"{CACHE_ROOT}/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_ROOT}/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_ROOT}/huggingface/datasets"

# ====== vLLM ======
os.environ["VLLM_CACHE_ROOT"] = f"{CACHE_ROOT}/vllm"

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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


LANGUAGE_TO_AFRIMGSM_CONFIG = {
    # English / French
    "EN": "eng",
    "ENG": "eng",
    "FR": "fra",
    "FRA": "fra",
    # African languages, ISO 639-3
    "AMH": "amh",
    "EWE": "ewe",
    "HAU": "hau",
    "IBO": "ibo",
    "KIN": "kin",
    "LIN": "lin",
    "LUG": "lug",
    "ORM": "orm",
    "SNA": "sna",
    "SOT": "sot",
    "SWA": "swa",
    "SW": "swa",  # backward compatibility
    "TWI": "twi",
    "VAI": "vai",
    "WOL": "wol",
    "XHO": "xho",
    "YOR": "yor",
    "ZUL": "zul",
}


POLYMATH_LANGUAGE_ALIASES = {
    "ENG": "EN",
    "FRA": "FR",
    "SWA": "SW",
}


def normalize_polymath_language(language: str) -> str:
    lang = language.upper()
    return POLYMATH_LANGUAGE_ALIASES.get(lang, lang)


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False

    try:
        pred_text = predicted.strip()
        gt_text = str(ground_truth).strip()

        if "$" not in pred_text and "\\boxed{" not in pred_text:
            pred_text = f"${pred_text}$"
        if "$" not in gt_text and "\\boxed{" not in gt_text:
            gt_text = f"${gt_text}$"

        pred_parsed = parse(pred_text)
        gt_parsed = parse(gt_text)

        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = str(ground_truth).replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def get_afrimgsm_ground_truth(example: dict) -> str:
    """
    AfriMGSM currently exposes answer_number, answer, and equation_solution in the viewer.
    Prefer answer_number because it is the final numeric answer.
    """
    if "answer_number" in example and example["answer_number"] is not None:
        return str(example["answer_number"])
    if "answer" in example and example["answer"] is not None:
        return str(example["answer"])
    raise KeyError(
        f"Cannot find ground-truth answer in example keys: {list(example.keys())}"
    )


def build_qwen3_prompt(
    tokenizer,
    question: str,
    language_code: str,
    enable_thinking: bool = True,
) -> str:
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

    if enable_thinking:
        return (
            base_prompt
            + "<|im_start|>assistant\n<think>\n"
            + LANGUAGE_PROMPTS[lang]["think_prefix"]
        )
    else:
        return base_prompt + "<|im_start|>assistant\n"


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = None,
    enable_thinking: bool = True,
):
    print(f"Loading model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        print(
            f"Auto-setting max_model_len to {max_model_len} for "
            f"{'thinking' if enable_thinking else 'non-thinking'} mode"
        )

    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        print(f"LoRA adapter path provided: {lora_adapter_path}")
        adapter_path = Path(lora_adapter_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            adapter_path = Path(lora_adapter_path) / "adapter_model.bin"

        if adapter_path.exists():
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}")
            print("Continuing with base model only...")
            lora_adapter_path = None

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("\n" + "=" * 70)
    print("MODEL DTYPE INFORMATION")
    print("=" * 70)
    print(f"vLLM Model Config dtype: {llm.llm_engine.model_config.dtype}")
    print(f"vLLM Model quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {llm.llm_engine.cache_config.cache_dtype}")
    print("=" * 70 + "\n")

    print("vLLM model loaded successfully!")
    return llm, tokenizer


def process_outputs_and_save(
    *,
    outputs,
    all_prompts,
    all_problems,
    all_gt_answers,
    all_question_ids,
    all_extra_fields,
    dataset_label: str,
    dataset_metadata: dict,
    tokenizer,
    llm,
    sampling_params,
    val_n: int,
    lora_request,
    output_file: str,
    base_model_name: str,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    max_new_tokens: int,
):
    total = 0
    formatted_count = 0
    results = []

    pass_at_n = 0
    total_correct_per_problem = 0

    print("\nProcessing results...")
    for idx, (
        output,
        prompt,
        problem,
        gt_answer,
        question_id,
        extra_fields,
    ) in enumerate(
        zip(
            outputs,
            all_prompts,
            all_problems,
            all_gt_answers,
            all_question_ids,
            all_extra_fields,
        )
    ):
        generations = []
        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []

        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text
            predicted_answer = extract_boxed_answer(generated_text)
            is_formatted = predicted_answer is not None
            is_correct = grade_answer(predicted_answer, gt_answer)

            generations.append(generated_text)
            predicted_answers.append(
                predicted_answer if predicted_answer else "[No boxed answer found]"
            )
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)

        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        majority_vote_correct = False
        if num_formatted > 0:
            formatted_predictions = [
                pred for pred, fmt in zip(predicted_answers, is_formatted_list) if fmt
            ]
            if formatted_predictions:
                most_common_answer = Counter(formatted_predictions).most_common(1)[0][0]
                majority_vote_correct = grade_answer(most_common_answer, gt_answer)

        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += val_n

        result = {
            "problem_id": question_id if question_id is not None else idx,
            "problem": problem,
            "prompt": prompt,
            "ground_truth": gt_answer,
            "val_n": val_n,
            "generations": [
                {
                    "predicted_answer": pred,
                    "full_generation": gen,
                    "correct": corr,
                    "formatted": fmt,
                }
                for pred, gen, corr, fmt in zip(
                    predicted_answers, generations, is_correct_list, is_formatted_list
                )
            ],
            "num_correct": num_correct,
            "pass_at_n": has_correct,
            "majority_vote_correct": majority_vote_correct,
            "predicted_answer": predicted_answers[0],
            "full_generation": generations[0],
            "correct": is_correct_list[0],
            "formatted": is_formatted_list[0],
        }
        result.update(extra_fields)
        results.append(result)

        format_rate = formatted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        status = "✓" if has_correct else "✗"
        print(
            f"{status} [{idx + 1}/{len(all_problems)}] "
            f"Pass@{val_n}: {current_pass_at_n:.1f}% | "
            f"Avg@{val_n}: {current_avg_at_n:.1f}% | "
            f"Formatted: {format_rate:.1f}%"
        )

    num_problems = len(all_problems)
    format_rate = formatted_count / total * 100 if total > 0 else 0.0
    pass_at_n_pct = pass_at_n / num_problems * 100 if num_problems > 0 else 0.0
    average_at_n_pct = total_correct_per_problem / total * 100 if total > 0 else 0.0
    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = (
        majority_vote_correct_count / num_problems * 100 if num_problems > 0 else 0.0
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_label}")
    for k, v in dataset_metadata.items():
        print(f"{k}: {v}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print("\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(
        f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})"
    )
    print(
        f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% "
        f"({majority_vote_correct_count}/{num_problems})"
    )
    print("\nFormatting:")
    print(f"  Formatted (boxed) answers: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "dataset": dataset_label.lower(),
            **dataset_metadata,
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def run_generation(
    *,
    llm,
    all_prompts,
    sampling_params,
    lora_request,
):
    print(f"\nRunning vLLM batch inference on {len(all_prompts)} problems...")

    print("\n" + "=" * 70)
    print("GENERATION DTYPE CHECK")
    print("=" * 70)
    print(f"Model dtype: {llm.llm_engine.model_config.dtype}")
    print(f"Quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {llm.llm_engine.cache_config.cache_dtype}")
    print(f"Using LoRA: {lora_request is not None}")
    if lora_request is not None:
        if lora_request.lora_path is None:
            raise ValueError(
                "LoRA request created but lora_local_path is None; "
                "lora weights are empty, might be issue with using zero3 + peft; try using zero2"
            )
        print(f"LoRA path: {lora_request.lora_path}")
    print("=" * 70 + "\n")

    if lora_request is not None:
        return llm.generate(
            all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True
        )
    return llm.generate(all_prompts, sampling_params, use_tqdm=True)


def evaluate_polymath(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int = None,
    output_file: str = None,
    lora_request=None,
    base_model_name: str = None,
    enable_thinking: bool = True,
    val_n: int = 6,
    language: str = "EN",
    polymath_split: str = "top",
):
    language = normalize_polymath_language(language)
    if language not in LANGUAGE_PROMPTS:
        raise ValueError(f"Unsupported language code: {language}")

    print(f"\n{'=' * 70}")
    print("EVALUATION CONFIGURATION")
    print(f"{'=' * 70}")
    print("Dataset: POLYMATH")
    print(f"Language: {language.upper()}")
    print(f"PolyMath split: {polymath_split}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Temperature: {temperature}")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"{'=' * 70}\n")

    dataset = load_dataset("Qwen/PolyMath", language.lower(), split=polymath_split)
    print(
        f"Loaded Qwen/PolyMath config={language.lower()} split={polymath_split} "
        f"with {len(dataset)} problems"
    )

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} problems with vLLM batch inference...")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    all_prompts = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []
    all_extra_fields = []

    for idx, example in enumerate(dataset):
        problem = example["question"]
        gt_answer = str(example["answer"])
        question_id = example.get("id", idx)

        prompt = build_qwen3_prompt(
            tokenizer=tokenizer,
            question=problem,
            language_code=language,
            enable_thinking=enable_thinking,
        )

        all_prompts.append(prompt)
        all_gt_answers.append(gt_answer)
        all_problems.append(problem)
        all_question_ids.append(question_id)
        all_extra_fields.append({})

    outputs = run_generation(
        llm=llm,
        all_prompts=all_prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    return process_outputs_and_save(
        outputs=outputs,
        all_prompts=all_prompts,
        all_problems=all_problems,
        all_gt_answers=all_gt_answers,
        all_question_ids=all_question_ids,
        all_extra_fields=all_extra_fields,
        dataset_label="polymath",
        dataset_metadata={
            "dataset_name": "Qwen/PolyMath",
            "language": language.upper(),
            "polymath_split": polymath_split,
        },
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
        val_n=val_n,
        lora_request=lora_request,
        output_file=output_file,
        base_model_name=base_model_name,
        enable_thinking=enable_thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        max_new_tokens=max_new_tokens,
    )


def evaluate_afrimgsm(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int = None,
    output_file: str = None,
    lora_request=None,
    base_model_name: str = None,
    enable_thinking: bool = True,
    val_n: int = 6,
    language: str = "SWA",
    afrimgsm_split: str = "test",
):
    lang = language.upper()
    if lang not in LANGUAGE_TO_AFRIMGSM_CONFIG:
        raise ValueError(
            f"Unsupported AfriMGSM language code: {lang}. "
            f"Supported: {sorted(LANGUAGE_TO_AFRIMGSM_CONFIG)}"
        )

    dataset_config = LANGUAGE_TO_AFRIMGSM_CONFIG[lang]

    print(f"\n{'=' * 70}")
    print("EVALUATION CONFIGURATION")
    print(f"{'=' * 70}")
    print("Dataset: AfriMGSM")
    print(f"Language: {lang}")
    print(f"AfriMGSM config: {dataset_config}")
    print(f"Split: {afrimgsm_split}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Temperature: {temperature}")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"{'=' * 70}\n")

    dataset = load_dataset("masakhane/afrimgsm", dataset_config, split=afrimgsm_split)
    print(
        f"Loaded masakhane/afrimgsm config={dataset_config} split={afrimgsm_split} "
        f"with {len(dataset)} problems"
    )

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} problems with vLLM batch inference...")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    all_prompts = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []
    all_extra_fields = []

    for idx, example in enumerate(dataset):
        problem = example["question"]
        gt_answer = get_afrimgsm_ground_truth(example)
        question_id = example.get("id", idx)

        prompt = build_qwen3_prompt(
            tokenizer=tokenizer,
            question=problem,
            language_code=lang,
            enable_thinking=enable_thinking,
        )

        all_prompts.append(prompt)
        all_gt_answers.append(gt_answer)
        all_problems.append(problem)
        all_question_ids.append(question_id)
        all_extra_fields.append(
            {
                "answer": example.get("answer", None),
                "answer_number": example.get("answer_number", None),
                "equation_solution": example.get("equation_solution", None),
            }
        )

    outputs = run_generation(
        llm=llm,
        all_prompts=all_prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    return process_outputs_and_save(
        outputs=outputs,
        all_prompts=all_prompts,
        all_problems=all_problems,
        all_gt_answers=all_gt_answers,
        all_question_ids=all_question_ids,
        all_extra_fields=all_extra_fields,
        dataset_label="afrimgsm",
        dataset_metadata={
            "dataset_name": "masakhane/afrimgsm",
            "dataset_config": dataset_config,
            "language": lang,
            "afrimgsm_split": afrimgsm_split,
        },
        tokenizer=tokenizer,
        llm=llm,
        sampling_params=sampling_params,
        val_n=val_n,
        lora_request=lora_request,
        output_file=output_file,
        base_model_name=base_model_name,
        enable_thinking=enable_thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        max_new_tokens=max_new_tokens,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on PolyMath or AfriMGSM with Qwen3 multilingual prompting"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="polymath",
        choices=["polymath", "afrimgsm"],
        help="Benchmark to evaluate on.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path or HF ID of base model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint directory with LoRA adapters. If not provided, base model only.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="EN",
        help=(
            "Language code. For PolyMath: EN, DE, ES, FR, JA, ZH, RU, SW, BN, TE, TH. "
            "For AfriMGSM: ENG, FRA, AMH, EWE, HAU, IBO, KIN, LIN, LUG, ORM, "
            "SNA, SOT, SWA, TWI, VAI, WOL, XHO, YOR, ZUL."
        ),
    )
    parser.add_argument(
        "--polymath_split",
        type=str,
        default="top",
        choices=["low", "medium", "high", "top"],
        help="Difficulty split for PolyMath.",
    )
    parser.add_argument(
        "--afrimgsm_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Split for masakhane/afrimgsm.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=38912,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable Qwen3 thinking mode.",
    )
    parser.add_argument(
        "--no_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Minimum probability threshold.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of problems to evaluate (None = all).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results JSON.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--val_n",
        type=int,
        default=6,
        help="Number of solutions to sample per problem.",
    )

    args = parser.parse_args()
    args.language = args.language.upper()

    if args.language not in LANGUAGE_PROMPTS:
        raise ValueError(
            f"Unsupported language code: {args.language}. "
            f"Supported prompt languages: {sorted(LANGUAGE_PROMPTS)}"
        )

    if (
        args.benchmark == "afrimgsm"
        and args.language not in LANGUAGE_TO_AFRIMGSM_CONFIG
    ):
        raise ValueError(
            f"Unsupported AfriMGSM language code: {args.language}. "
            f"Supported AfriMGSM languages: {sorted(LANGUAGE_TO_AFRIMGSM_CONFIG)}"
        )

    if args.benchmark == "polymath":
        polymath_lang = normalize_polymath_language(args.language)
        if polymath_lang != args.language:
            print(
                f"[INFO] Normalizing PolyMath language {args.language} -> {polymath_lang}"
            )
            args.language = polymath_lang

    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"\n{'=' * 70}")
            print("ERROR: Checkpoint directory does not exist")
            print(f"{'=' * 70}")
            print(f"Provided checkpoint directory: {args.checkpoint_dir}")
            print("This directory does not exist.")
            print(f"{'=' * 70}\n")
            raise SystemExit(1)

    if args.output_file is None:
        if args.benchmark == "polymath":
            split_name = args.polymath_split
        else:
            split_name = args.afrimgsm_split

        parts = [
            "eval_results",
            args.benchmark,
            args.language.lower(),
            split_name,
            Path(args.base_model).name,
        ]

        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            parts += [checkpoint_path.parent.name, checkpoint_path.name]

        parts += [
            "thinking" if args.enable_thinking else "nonthinking",
            f"temp{args.temperature}",
            f"valn{args.val_n}",
        ]
        args.output_file = str(Path("eval_results") / ("_".join(parts) + ".json"))

    print(f"Results will be saved to: {args.output_file}")

    print("\n" + "=" * 70)
    print("QWEN3 MATH EVALUATION")
    print("=" * 70)
    print(f"Benchmark: {args.benchmark}")
    print(f"Language: {args.language}")

    if args.benchmark == "polymath":
        print(f"PolyMath split: {args.polymath_split}")
    else:
        print(f"AfriMGSM split: {args.afrimgsm_split}")
        print(f"AfriMGSM config: {LANGUAGE_TO_AFRIMGSM_CONFIG.get(args.language)}")

    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint_dir or 'None (base model only)'}")
    print(f"Thinking Mode: {'ENABLED ✓' if args.enable_thinking else 'DISABLED'}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Num samples: {args.num_samples or 'All'}")
    print(f"Val-N: {args.val_n}")
    print(f"Output file: {args.output_file}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 70 + "\n")

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = None
    if args.checkpoint_dir is not None:
        try:
            from vllm.lora.request import LoRARequest

            adapter_safetensors = (
                Path(args.checkpoint_dir) / "adapter_model.safetensors"
            )
            adapter_bin = Path(args.checkpoint_dir) / "adapter_model.bin"

            if adapter_safetensors.exists() or adapter_bin.exists():
                lora_request = LoRARequest("checkpoint_lora", 1, args.checkpoint_dir)
                print(f"✓ Successfully created LoRA request for: {args.checkpoint_dir}")
            else:
                print(
                    f"Warning: No LoRA adapter weights found at {args.checkpoint_dir}"
                )
                print("Continuing with base model only...")
        except ImportError:
            print("Warning: Could not import LoRARequest. Running without LoRA.")
        except Exception as e:
            print(f"Warning: Could not create LoRA request: {e}")
            print("Continuing without LoRA.")

    if args.benchmark == "polymath":
        average_at_n_pct, results = evaluate_polymath(
            llm,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            num_samples=args.num_samples,
            output_file=args.output_file,
            lora_request=lora_request,
            base_model_name=args.base_model,
            enable_thinking=args.enable_thinking,
            val_n=args.val_n,
            language=args.language,
            polymath_split=args.polymath_split,
        )
    elif args.benchmark == "afrimgsm":
        average_at_n_pct, results = evaluate_afrimgsm(
            llm,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            num_samples=args.num_samples,
            output_file=args.output_file,
            lora_request=lora_request,
            base_model_name=args.base_model,
            enable_thinking=args.enable_thinking,
            val_n=args.val_n,
            language=args.language,
            afrimgsm_split=args.afrimgsm_split,
        )
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
