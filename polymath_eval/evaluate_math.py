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

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

LANGUAGE_PROMPTS = {
    "EN": {
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
}


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

    total = 0
    formatted_count = 0
    results = []

    pass_at_n = 0
    total_correct_per_problem = 0

    all_prompts = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []

    for example in dataset:
        problem = example["question"]
        gt_answer = str(example["answer"])
        question_id = example.get("id", None)

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
        outputs = llm.generate(
            all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True
        )
    else:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    print("\nProcessing results...")
    for idx, (output, prompt, problem, gt_answer, question_id) in enumerate(
        zip(outputs, all_prompts, all_problems, all_gt_answers, all_question_ids)
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
        results.append(result)

        format_rate = formatted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        status = "✓" if has_correct else "✗"
        print(
            f"{status} [{idx + 1}/{len(dataset)}] "
            f"Pass@{val_n}: {current_pass_at_n:.1f}% | "
            f"Avg@{val_n}: {current_avg_at_n:.1f}% | "
            f"Formatted: {format_rate:.1f}%"
        )

    num_problems = len(dataset)
    format_rate = formatted_count / total * 100
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100
    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print("Dataset: POLYMATH")
    print(f"Language: {language.upper()}")
    print(f"Split: {polymath_split}")
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
            "dataset": "polymath",
            "language": language.upper(),
            "polymath_split": polymath_split,
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on PolyMath with Qwen3 multilingual prompting"
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
        help="PolyMath language code, e.g. EN, DE, ES, JA, ZH.",
    )
    parser.add_argument(
        "--polymath_split",
        type=str,
        default="top",
        choices=["low", "medium", "high", "top"],
        help="Difficulty split for PolyMath.",
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
        raise ValueError(f"Unsupported language code: {args.language}")

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
        parts = [
            "eval_results",
            "polymath",
            args.language.lower(),
            args.polymath_split,
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
    print("QWEN3 POLYMATH EVALUATION")
    print("=" * 70)
    print(f"Language: {args.language}")
    print(f"PolyMath split: {args.polymath_split}")
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

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
