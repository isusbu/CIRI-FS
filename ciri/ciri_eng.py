import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ciri.ciri_logger import logger
from ciri.ciri_runner import ciri_runner
from ciri.pre_processing.shot.shot_selection import ShotSelection
from ciri.pre_processing.code_retrieval.config_usage_retriever import ConfigUsageRetriever
from ciri.pre_processing.code_retrieval.config_usage_retriever import ConfigUsageRetriever as InfoProvider

class CiriEngineError(Exception):
    """Base exception class for Ciri Engine errors."""
    pass

class DirectoryError(CiriEngineError):
    """Raised when there are issues with directory operations."""
    pass

class ModelLoadError(CiriEngineError):
    """Raised when there are issues loading the model."""
    pass

class ArgumentError(CiriEngineError):
    """Raised when there are issues with command line arguments."""
    pass

def create_directory(directory: Union[str, Path], recreate: bool = True) -> Path:
    """
    Create a directory, optionally removing it first if it exists.
    
    Args:
        directory: Path to the directory to create
        recreate: If True, removes existing directory before creation
        
    Returns:
        Path object of created directory
        
    Raises:
        DirectoryError: If directory operations fail
    """
    try:
        directory_path = Path(directory)
        if recreate and directory_path.exists():
            shutil.rmtree(directory_path)
        directory_path.mkdir(parents=True, exist_ok=not recreate)
        return directory_path
    except (OSError, shutil.Error) as e:
        raise DirectoryError(f"Failed to create directory {directory}: {str(e)}")

def update_logger_handler(log_path: Union[str, Path]) -> None:
    """
    Update the logger's file handler to write to a new location.
    
    Args:
        log_path: Path to the new log file
        
    Raises:
        OSError: If unable to create or access log file
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    
    new_handler = logging.FileHandler(log_path)
    new_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(new_handler)

def load_language_model(checkpoint: str) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Load the specified language model and tokenizer.
    Supports both GPU (CUDA) and CPU execution.
    
    Args:
        checkpoint: Model checkpoint identifier
        
    Returns:
        Tuple of (model, tokenizer), both None for API-based models
        
    Raises:
        ModelLoadError: If model loading fails
    """
    # API-based models 
    if checkpoint in [
        "gpt-3.5-turbo-0125",
        "gpt-4-0125-preview",
        "claude-3-opus-20240228",
        "claude-3-sonnet-20240229"
    ]:
        return None, None

    try:
        # Detect device instead of requiring CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use of appropriate dtype for each device
        # bfloat16 is optimal for GPU, float32 required for CPU
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        logger.info(f"Using device: {device.upper()}")
        logger.info(f"Using dtype: {dtype}")

        if checkpoint.startswith("deepseek"):
            
            # Use device-aware device_map
            full_checkpoint = f"deepseek-ai/{checkpoint}"
            logger.info(f"Loading DeepSeek model: {full_checkpoint}")
            # updated using  BitsAndBytesConfig for making run in google colab
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                full_checkpoint,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                full_checkpoint,
                trust_remote_code=True,
                padding_side='left'
            )

        elif checkpoint.startswith("Qwen"):
            from transformers import BitsAndBytesConfig

            full_checkpoint = "Qwen/Qwen2.5-Coder-7B-Instruct"

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                full_checkpoint,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(
                full_checkpoint,
                trust_remote_code=True,
                padding_side="left"
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

        elif checkpoint.startswith("CodeLLaMa"):
            full_checkpoint = f"codellama/{checkpoint}"
            logger.info(f"Loading CodeLlama model: {full_checkpoint}")
            
            model = AutoModelForCausalLM.from_pretrained(
                full_checkpoint,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=dtype
            )
            tokenizer = AutoTokenizer.from_pretrained(
                full_checkpoint,
                padding_side='left'
            )

        else:
            raise ValueError(f"Unsupported model checkpoint: {checkpoint}")

        # Removed hardcoded model.to("cuda") — device_map handles placement
        model.eval()
        logger.info(f"Model loaded successfully on {device.upper()}!")
        return model, tokenizer

    except Exception as e:
        raise ModelLoadError(f"Failed to load model {checkpoint}: {str(e)}")

def process_single_file(
    args: argparse.Namespace,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None
) -> None:
    """
    Process a single configuration file through the Ciri engine.
    
    Args:
        args: Command line arguments
        input_path: Path to input file
        output_path: Path to output file
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer
        
    Raises:
        FileNotFoundError: If input file does not exist
        OSError: If unable to read input file or write output file
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    logger.debug(f"Processing file: {input_path}")
    input_content = input_path.read_text()
    
    if args.validconfig_shot_num + args.misconfig_shot_num > 0:
        shot_selector = ShotSelection(args, input_content)
        input_content = shot_selector.select() + input_content

    if args.read_code:
        if not args.read_code_loc:
            raise ArgumentError("read_code_loc must be specified when read_code is enabled")
            
        info_provider = InfoProvider(
            args.system,
            args.read_code_loc,
            args.file_format,
            args.language
        )
        input_content += info_provider.generate(input_content)
    
    logger.debug(f"Prepared content:\n{input_content}")
    ciri_runner(args, input_content, output_path, model, tokenizer)

def process_files(args: argparse.Namespace) -> None:
    """
    Process input files through the Ciri engine.
    
    Args:
        args: Command line arguments
        
    Raises:
        CiriEngineError: If processing fails
        FileNotFoundError: If input path does not exist
    """
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if input_path.is_dir():
        output_path = create_directory(output_path)
        model, tokenizer = None, None
        
        if hasattr(args, 'model'):
            model, tokenizer = load_language_model(args.model)
            
        for file_path in input_path.glob('*'):
            if file_path.is_file():
                output_file = output_path / file_path.name
                update_logger_handler(output_file)
                process_single_file(args, file_path, output_file, model, tokenizer)
            
    elif input_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        model, tokenizer = load_language_model(args.model)
        update_logger_handler(output_path)
        process_single_file(
            args,
            input_path,
            output_path,
            model,
            tokenizer
        )
    else:
        raise CiriEngineError(f"Invalid input path: {input_path}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        Parsed command line arguments
        
    Raises:
        ArgumentError: If argument validation fails
    """
    parser = argparse.ArgumentParser(
        description="Ciri Engine - Configuration Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--input_path", required=True, type=str,
                         help="Path to input file or directory")
    required.add_argument("--output_path", required=True, type=str,
                         help="Path to output file or directory")
    required.add_argument("--model", required=True, type=str,
                         choices=[
                             "gpt-3.5-turbo-0125",
                             "gpt-4-0125-preview",
                             "claude-3-opus-20240228",
                             "claude-3-sonnet-20240229",
                             "CodeLLaMa-7b-Instruct-hf",
                             "CodeLLaMa-13b-Instruct-hf",
                             "CodeLLaMa-34b-Instruct-hf",
                             "deepseek-coder-6.7b-instruct", "Qwen2.5-Coder-7B-Instruct"
                         ],
                         help="Name of the model to use")
    required.add_argument("--system", required=True, type=str,
                         help="Software System name for processing")
    required.add_argument("--version", required=True, type=str,
                         help="System version for processing")
    
    # Optional arguments
    parser.add_argument("--validconfig_shot_num", type=int, default=1,
                       choices=range(0, 6),
                       help="Number of valid configuration shots")
    parser.add_argument("--misconfig_shot_num", type=int, default=3,
                       choices=range(0, 6),
                       help="Number of misconfiguration shots")
    parser.add_argument("--shot_selection", type=str, default="random",
                       choices=["random", "similarity"],
                       help="Shot selection strategy")
    parser.add_argument("--file_format", type=str, default="xml",
                       choices=["xml", "yaml", "properties", "conf"],
                       help="Configuration file format")
    parser.add_argument("--language", type=str, default="java",
                       choices=["java", "python", "cpp"],
                       help="Project programming language")
    parser.add_argument("--read_code", action="store_true",
                       help="Enable code reading for configuration")
    parser.add_argument("--read_code_loc", type=str,
                       help="Code repository path (required if read_code is enabled)")
    parser.add_argument("--shot_system", type=str,
                       help="Shot system identifier")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()
    
    # Validate arguments
    if args.read_code and not args.read_code_loc:
        parser.error("--read_code_loc is required when --read_code is enabled")
        
    if args.validconfig_shot_num + args.misconfig_shot_num > 5:
        parser.error("Total number of shots cannot exceed 5")
        
    logger.debug(f"Parsed arguments: {args}")
    return args

def main() -> None:
    """
    Main entry point for the Ciri Engine.
    
    Sets up logging and executes the configuration analysis pipeline.
    """
    args = None
    try:
        args = parse_arguments()
        
        # Configure logging based on verbosity
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logger.setLevel(log_level)
        
        process_files(args)
        
    except Exception as e:
        # Guard against args being None if parse_arguments() itself fails
        verbose = args.verbose if args is not None else False
        logger.error(f"Ciri Engine failed: {str(e)}", exc_info=verbose)
        raise
    finally:
        # Clean up logging handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
    main()
