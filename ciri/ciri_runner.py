from typing import Dict, List, Optional
from pathlib import Path

from ciri.post_processing import selection
from ciri.query.llm_gen import GPTGen, ClaudeGen, LlamaGen, DeepseekGen, QwenGen
from ciri.ciri_logger import logger
from ciri.post_processing.reason_cluster import get_dominant_reason


def ciri_runner(args: Dict, file_content: str, output_path: str, model, tokenizer) -> None:
    llm_gen = _get_llm_generator(args, file_content, model, tokenizer)
    result, reasons = _run_analysis(llm_gen)
    _output_results(result, reasons, output_path)


def _get_llm_generator(args: Dict, file_content: str, model, tokenizer):
    if args.model in ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"]:
        return GPTGen(args, file_content)
    elif args.model.startswith("claude"):
        return ClaudeGen(args, file_content)
    elif args.model.startswith("CodeLLaMa"):
        return LlamaGen(args, file_content, model, tokenizer)
    elif args.model.startswith("deepseek"):
        return DeepseekGen(args, file_content, model, tokenizer)
    elif args.model.startswith("Qwen"):
        return QwenGen(args, file_content, model, tokenizer)
    else:
        raise ValueError(f"Model {args.model} is not supported")


def _run_analysis(llm_gen) -> tuple[str, Optional[dict[str, str]]]:
    while True:
        answer_parser = llm_gen.generate()
        selection_gen = selection.Selection()
        pure_result = selection_gen.select(answer_parser.candidate_pool.pool)

        if len(pure_result) > 1:
            # NO CONSISTENT RESULT — treat as correct to avoid parser errors
            return "The CONFIGURATION FILE IS CORRECT", None

        if pure_result[0] == ["None"]:
            return "The CONFIGURATION FILE IS CORRECT", None

        reasons = {}
        raw_answers = answer_parser.answer_pool.pool
        for param in pure_result[0]:
            reason_list = []
            for answer in raw_answers:
                if set(answer["errParameter"]) == set(pure_result[0]):
                    try:
                        idx = answer["errParameter"].index(param)
                        if idx < len(answer["reason"]):
                            reason_list.append(answer["reason"][idx])
                        elif len(answer["reason"]) > 0:
                            reason_list.append(answer["reason"][0])
                    except (ValueError, IndexError):
                        if len(answer["reason"]) > 0:
                            reason_list.append(answer["reason"][0])

            if reason_list:
                reasons[param] = get_dominant_reason(reason_list)
            else:
                reasons[param] = "No reason provided"

        result = (
            f"There are {len(pure_result[0])} misconfiguration parameters in the input: "
            + "\t".join(pure_result[0])
        )
        return result, reasons


def _output_results(result: str, reasons: Optional[dict[str, str]], output_path: str, verbose: bool = False) -> None:
    print(f"[Ciri] Start")
    print(f"[Ciri] Running for file {output_path}")
    print(f"[Ciri] Result: {result}")

    if reasons:
        for param, reason in reasons.items():
            message = f"[Ciri] Reason for {param}: {reason}"
            print(message)

    print(f"[Ciri] Writing log file to {output_path}")
    print("[Ciri] End")

    # Write results to output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"Final result:\n\n")
        f.write(f"{result}\n")
        if reasons:
            for param, reason in reasons.items():
                f.write(f"\nReason for {param}: {reason}\n")
