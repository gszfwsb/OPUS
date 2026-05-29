#!/usr/bin/env python3
"""
Offline LightEval evaluation for NanoGPT checkpoints.

This script provides a complete offline evaluation solution using the lighteval framework,
adapted for your custom GPT models trained with train.py.

Features:
- 100% offline: Uses pre-downloaded datasets from opencompass/data/
- Custom model loading: Loads your .pt checkpoints directly
- Supports all model types: gpt2, gpt2-medium, gpt2-large, gpt2-xl

Usage:
    python run_lighteval_offline.py --checkpoint logs/experiment/state_step010000.pt --model_type gpt2-xl

Author: Generated for Muon-Pretrain project
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import multiprocessing as mp
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np

# Set offline mode BEFORE importing anything from HuggingFace
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "OPUS"))

import tiktoken
from model import GPT


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    checkpoint_path: str
    model_type: str = "gpt2-xl"
    hf_compatible: bool = False
    max_seq_len: int = 2048  # Use shorter context for eval speed
    batch_size: int = 64  # Increased for better GPU utilization (H200: 143GB VRAM)
    device: str = "cuda"
    output_dir: str = "lighteval_results"
    tasks: List[str] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["hellaswag", "winogrande", "piqa", "siqa", "arc_easy", "arc_challenge", "mmlu"]


# =============================================================================
# Dataset Loaders (Offline)
# =============================================================================

def load_hellaswag_offline(data_dir: Path) -> List[Dict]:
    """Load HellaSwag from local JSONL files (OpenCompass format)"""
    hellaswag_dir = data_dir / "hellaswag"
    data_file = hellaswag_dir / "hellaswag.jsonl"
    
    if not data_file.exists():
        print(f"Warning: HellaSwag data not found at {data_file}")
        return []
    
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            # OpenCompass format: query, choices, gold
            if 'query' in item:
                data.append({
                    'query': item.get('query', ''),
                    'choices': item.get('choices', []),
                    'label': int(item.get('gold', -1)),
                    'format': 'opencompass',
                })
            # Original lighteval format: ctx_a, ctx_b, endings, label
            else:
                data.append({
                    'ctx': item.get('ctx', ''),
                    'ctx_a': item.get('ctx_a', ''),
                    'ctx_b': item.get('ctx_b', ''),
                    'activity_label': item.get('activity_label', ''),
                    'endings': item.get('endings', []),
                    'label': int(item.get('label', -1)) if item.get('label', '') != '' else -1,
                    'format': 'lighteval',
                })
    return data


def load_arc_offline(data_dir: Path, subset: str = "ARC-c") -> List[Dict]:
    """Load ARC dataset from local files (OpenCompass format)"""
    arc_dir = data_dir / "ARC" / subset
    
    # Try different file naming conventions
    test_files = [
        arc_dir / "ARC-Challenge-Test.jsonl" if subset == "ARC-c" else arc_dir / "ARC-Easy-Test.jsonl",
        arc_dir / "test.jsonl",
    ]
    
    test_file = None
    for tf in test_files:
        if tf.exists():
            test_file = tf
            break
    
    if test_file is None:
        print(f"Warning: ARC-{subset} data not found in {arc_dir}")
        return []
    
    print(f"Loading ARC data from: {test_file}")
    
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # OpenCompass format: question.stem, question.choices, answerKey
            question_data = item.get('question', {})
            if isinstance(question_data, dict):
                question = question_data.get('stem', '')
                choices_list = question_data.get('choices', [])
                choice_texts = [c.get('text', '') for c in choices_list]
                choice_labels = [c.get('label', '') for c in choices_list]
            else:
                question = item.get('question', '')
                choices_list = item.get('choices', [])
                choice_texts = [c.get('text', '') for c in choices_list] if choices_list else []
                choice_labels = [c.get('label', '') for c in choices_list] if choices_list else []
            
            # Find gold index
            answer_key = item.get('answerKey', 'A')
            gold_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
            
            data.append({
                'question': question,
                'choices': choice_texts,
                'gold_index': gold_idx,
            })
    return data


def load_piqa_offline(data_dir: Path) -> List[Dict]:
    """Load PiQA from local files"""
    piqa_dir = data_dir / "piqa"
    data_file = piqa_dir / "dev.jsonl"
    labels_file = piqa_dir / "dev-labels.lst"
    
    if not data_file.exists():
        print(f"Warning: PiQA data not found at {data_file}")
        return []
    
    # Load labels
    labels = []
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels = [int(l.strip()) for l in f if l.strip()]
    
    data = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            label = labels[i] if i < len(labels) else 0
            data.append({
                'goal': item.get('goal', ''),
                'sol1': item.get('sol1', ''),
                'sol2': item.get('sol2', ''),
                'label': label,
            })
    return data


def load_siqa_offline(data_dir: Path) -> List[Dict]:
    """Load SocialIQA from local files"""
    siqa_dir = data_dir / "siqa"
    data_file = siqa_dir / "dev.jsonl"
    labels_file = siqa_dir / "dev-labels.lst"
    
    if not data_file.exists():
        print(f"Warning: SocialIQA data not found at {data_file}")
        return []
    
    # Load labels
    labels = []
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            labels = [int(l.strip()) for l in f if l.strip()]
    
    data = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            label = labels[i] if i < len(labels) else 1
            data.append({
                'context': item.get('context', ''),
                'question': item.get('question', ''),
                'answerA': item.get('answerA', ''),
                'answerB': item.get('answerB', ''),
                'answerC': item.get('answerC', ''),
                'label': label,  # 1-indexed
            })
    return data


def load_winogrande_offline(data_dir: Path) -> List[Dict]:
    """Load Winogrande from local files (OpenCompass format)"""
    wino_dir = data_dir / "winogrande"
    
    # Try different file locations
    data_files = [
        wino_dir / "dev.jsonl",
        wino_dir / "winogrande_xl" / "validation.jsonl",
    ]
    
    data_file = None
    for df in data_files:
        if df.exists():
            data_file = df
            break
    
    if data_file is None:
        print(f"Warning: Winogrande data not found in {wino_dir}")
        return []
    
    print(f"Loading Winogrande from: {data_file}")
    
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'sentence': item.get('sentence', ''),
                'option1': item.get('option1', ''),
                'option2': item.get('option2', ''),
                'answer': str(item.get('answer', '1')),  # Ensure string
            })
    return data


def load_commonsenseqa_offline(data_dir: Path) -> List[Dict]:
    """Load CommonsenseQA from local files"""
    csqa_dir = data_dir / "commonsenseqa"
    data_file = csqa_dir / "dev_rand_split.jsonl"
    
    if not data_file.exists():
        print(f"Warning: CommonsenseQA data not found at {data_file}")
        return []
    
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Extract choices
            choices = item.get('question', {}).get('choices', [])
            choice_texts = [c.get('text', '') for c in choices]
            choice_labels = [c.get('label', '') for c in choices]
            
            answer_key = item.get('answerKey', 'A')
            gold_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
            
            data.append({
                'question': item.get('question', {}).get('stem', ''),
                'choices': choice_texts,
                'gold_index': gold_idx,
            })
    return data


def load_openbookqa_offline(data_dir: Path) -> List[Dict]:
    """Load OpenBookQA from local files"""
    obqa_dir = data_dir / "openbookqa" / "Main"
    test_file = obqa_dir / "test.jsonl"
    
    if not test_file.exists():
        print(f"Warning: OpenBookQA data not found at {test_file}")
        return []
    
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Extract choices
            choices = item.get('question', {}).get('choices', [])
            choice_texts = [c.get('text', '') for c in choices]
            choice_labels = [c.get('label', '') for c in choices]
            
            answer_key = item.get('answerKey', 'A')
            gold_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
            
            data.append({
                'question_stem': item.get('question', {}).get('stem', ''),
                'choices': choice_texts,
                'gold_index': gold_idx,
            })
    return data


def load_mmlu_offline(data_dir: Path, subject: str = None) -> List[Dict]:
    """
    Load MMLU from local files.
    
    File format: {subject}_test.csv
    CSV format: question,choice_A,choice_B,choice_C,choice_D,answer_letter
    
    This matches the lighteval/mmlu HuggingFace dataset format used by lighteval_task.py
    """
    mmlu_dir = data_dir / "mmlu" / "test"
    
    if not mmlu_dir.exists():
        print(f"Warning: MMLU data not found at {mmlu_dir}")
        return []
    
    data = []
    
    # Get all subjects or specific one
    if subject:
        subjects = [subject]
    else:
        # Files are named {subject}_test.csv
        subjects = [f.stem.replace('_test', '') for f in mmlu_dir.glob("*_test.csv")]
    
    print(f"Loading MMLU from {len(subjects)} subjects...")
    
    for subj in subjects:
        # Try both naming conventions: {subject}_test.csv and {subject}.csv
        csv_file = mmlu_dir / f"{subj}_test.csv"
        if not csv_file.exists():
            csv_file = mmlu_dir / f"{subj}.csv"
        if not csv_file.exists():
            continue
            
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    question = row[0]
                    choices = row[1:5]
                    answer = row[5].strip()
                    
                    # Convert answer letter to index (matches lighteval format)
                    # lighteval uses line["answer"] which is the integer index
                    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                    gold_idx = answer_map.get(answer.upper(), 0)
                    
                    data.append({
                        'question': question,
                        'subject': subj,
                        'choices': choices,
                        'answer': gold_idx,  # Match lighteval field name
                        'gold_index': gold_idx,  # Keep for backward compatibility
                    })
    
    print(f"Loaded {len(data)} MMLU samples")
    return data


# =============================================================================
# Model Wrapper
# =============================================================================

class NanoGPTEvaluator:
    """
    Evaluation wrapper for NanoGPT checkpoints.
    Provides methods for perplexity-based evaluation (like lighteval).
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load tokenizer (tiktoken works offline)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.pad_id = 50256  # EOS as padding
        self.eos_id = 50256
        
        # Load model
        print(f"Loading model: {config.model_type}")
        self.model = GPT.from_model_type(
            config.model_type,
            max_seq_len=config.max_seq_len,
            hf_compatible=config.hf_compatible
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {config.checkpoint_path}")
        ckpt = torch.load(config.checkpoint_path, map_location='cpu')
        
        # Clean state dict (remove _orig_mod prefix from torch.compile)
        sd = ckpt.get('model', ckpt)
        clean_sd = {}
        for k, v in sd.items():
            new_k = k.replace('_orig_mod.', '').replace('module.', '')
            clean_sd[new_k] = v
        
        self.model.load_state_dict(clean_sd, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Disable softcapping for evaluation consistency
        if hasattr(self.model, 'disable_logit_squash'):
            self.model.disable_logit_squash = False  # Keep softcapping ON
        
        print(f"Model loaded: {self.model.get_num_params()/1e6:.1f}M parameters")
    
    def encode(self, text: str) -> List[int]:
        """Tokenize text"""
        return self.tokenizer.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids"""
        return self.tokenizer.decode(ids)
    
    @torch.no_grad()
    def get_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for each position.
        
        Args:
            input_ids: [batch, seq_len] tensor of token ids
            
        Returns:
            [batch, seq_len, vocab_size] tensor of log probabilities
        """
        logits = self.model(input_ids, target_seq=None, sliding_window_num_blocks=None)
        return F.log_softmax(logits.float(), dim=-1)
    
    @torch.no_grad()
    def score_choices(self, context: str, choices: List[str]) -> Tuple[int, List[float]]:
        """
        Score multiple choice options using log-likelihood.
        Returns the predicted choice index and scores for all choices.
        
        This mimics lighteval's loglikelihood_acc evaluation.
        """
        context_ids = self.encode(context)
        
        scores = []
        for choice in choices:
            choice_ids = self.encode(choice)
            full_ids = context_ids + choice_ids
            
            # Truncate if needed
            if len(full_ids) > self.config.max_seq_len:
                full_ids = full_ids[-self.config.max_seq_len:]
            
            input_tensor = torch.tensor([full_ids], device=self.device)
            log_probs = self.get_logprobs(input_tensor)
            
            # Calculate log-likelihood for the choice tokens only
            # We score P(choice_token | context + previous_choice_tokens)
            choice_start = len(context_ids) - 1  # -1 because log_probs[i] predicts token i+1
            if choice_start < 0:
                choice_start = 0
            
            total_logprob = 0.0
            for i, token_id in enumerate(choice_ids):
                pos = choice_start + i
                if pos < log_probs.shape[1]:
                    total_logprob += log_probs[0, pos, token_id].item()
            
            # Normalize by length (optional, can be toggled)
            # avg_logprob = total_logprob / len(choice_ids) if choice_ids else 0.0
            scores.append(total_logprob)
        
        # Return predicted index (highest score) and all scores
        pred_idx = np.argmax(scores)
        return int(pred_idx), scores
    
    @torch.no_grad()
    def score_choices_normalized(self, context: str, choices: List[str]) -> Tuple[int, List[float]]:
        """
        Score with length normalization (loglikelihood_acc_norm).
        """
        context_ids = self.encode(context)
        
        scores = []
        for choice in choices:
            choice_ids = self.encode(choice)
            full_ids = context_ids + choice_ids
            
            if len(full_ids) > self.config.max_seq_len:
                full_ids = full_ids[-self.config.max_seq_len:]
            
            input_tensor = torch.tensor([full_ids], device=self.device)
            log_probs = self.get_logprobs(input_tensor)
            
            choice_start = len(context_ids) - 1
            if choice_start < 0:
                choice_start = 0
            
            total_logprob = 0.0
            for i, token_id in enumerate(choice_ids):
                pos = choice_start + i
                if pos < log_probs.shape[1]:
                    total_logprob += log_probs[0, pos, token_id].item()
            
            # Normalize by number of tokens
            avg_logprob = total_logprob / len(choice_ids) if choice_ids else float('-inf')
            scores.append(avg_logprob)
        
        pred_idx = np.argmax(scores)
        return int(pred_idx), scores
    
    @torch.no_grad()
    def score_batch(self, samples: List[Tuple[str, List[str]]]) -> List[Tuple[int, int, List[float], List[float]]]:
        """
        Batch scoring for multiple samples at once.
        
        Args:
            samples: List of (context, choices) tuples
            
        Returns:
            List of (pred_idx, pred_idx_norm, scores, scores_norm) tuples
        """
        if not samples:
            return []
        
        # Prepare all sequences for batching
        all_sequences = []  # [(sample_idx, choice_idx, full_ids, context_len, choice_len)]
        
        for sample_idx, (context, choices) in enumerate(samples):
            context_ids = self.encode(context)
            for choice_idx, choice in enumerate(choices):
                choice_ids = self.encode(choice)
                full_ids = context_ids + choice_ids
                
                if len(full_ids) > self.config.max_seq_len:
                    # Truncate from the left
                    offset = len(full_ids) - self.config.max_seq_len
                    full_ids = full_ids[-self.config.max_seq_len:]
                    context_len = max(0, len(context_ids) - offset)
                else:
                    context_len = len(context_ids)
                
                all_sequences.append({
                    'sample_idx': sample_idx,
                    'choice_idx': choice_idx,
                    'full_ids': full_ids,
                    'context_len': context_len,
                    'choice_ids': choice_ids,
                })
        
        # Process in batches
        batch_size = self.config.batch_size
        all_results = []
        
        for batch_start in range(0, len(all_sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(all_sequences))
            batch = all_sequences[batch_start:batch_end]
            
            # Pad sequences to same length
            max_len = max(len(seq['full_ids']) for seq in batch)
            padded_ids = []
            for seq in batch:
                padding = [self.pad_id] * (max_len - len(seq['full_ids']))
                padded_ids.append(seq['full_ids'] + padding)
            
            # Forward pass
            input_tensor = torch.tensor(padded_ids, device=self.device)
            log_probs = self.get_logprobs(input_tensor)
            
            # Extract scores for each sequence in batch
            for i, seq in enumerate(batch):
                context_len = seq['context_len']
                choice_ids = seq['choice_ids']
                choice_start = context_len - 1
                if choice_start < 0:
                    choice_start = 0
                
                total_logprob = 0.0
                for j, token_id in enumerate(choice_ids):
                    pos = choice_start + j
                    if pos < len(seq['full_ids']):
                        total_logprob += log_probs[i, pos, token_id].item()
                
                all_results.append({
                    'sample_idx': seq['sample_idx'],
                    'choice_idx': seq['choice_idx'],
                    'score': total_logprob,
                    'score_norm': total_logprob / len(choice_ids) if choice_ids else float('-inf'),
                })
        
        # Group results by sample
        num_samples = len(samples)
        sample_results = [[] for _ in range(num_samples)]
        for r in all_results:
            sample_results[r['sample_idx']].append(r)
        
        # Compute final predictions
        final_results = []
        for sample_idx, results in enumerate(sample_results):
            # Sort by choice_idx to maintain order
            results.sort(key=lambda x: x['choice_idx'])
            scores = [r['score'] for r in results]
            scores_norm = [r['score_norm'] for r in results]
            pred_idx = int(np.argmax(scores))
            pred_idx_norm = int(np.argmax(scores_norm))
            final_results.append((pred_idx, pred_idx_norm, scores, scores_norm))
        
        return final_results


# =============================================================================
# Multi-GPU Support
# =============================================================================

def evaluate_data_shard(gpu_id: int, data_shard: List[Dict], task_name: str, 
                       checkpoint_path: str, model_type: str, hf_compatible: bool,
                       max_seq_len: int, batch_size: int) -> Dict:
    """
    Worker function to evaluate a data shard on a specific GPU.
    Called by multiprocessing pool.
    
    Note: This function runs in a separate process, so we need to set up
    the CUDA device BEFORE importing torch/CUDA modules.
    """
    import os
    # Set CUDA device BEFORE any CUDA initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Now safe to import CUDA-related modules
    import torch
    print(f"[GPU {gpu_id}] Worker started with {len(data_shard)} samples for task '{task_name}'")
    
    # Create evaluator on this GPU
    config = EvalConfig(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        hf_compatible=hf_compatible,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device="cuda",
    )
    
    evaluator = NanoGPTEvaluator(config)
    
    # Run evaluation based on task
    result = None
    if task_name == "hellaswag":
        result = evaluate_hellaswag_core(evaluator, data_shard)
    elif task_name == "arc_easy" or task_name == "arc_challenge":
        result = evaluate_arc_core(evaluator, data_shard)
    elif task_name == "piqa":
        result = evaluate_piqa_core(evaluator, data_shard)
    elif task_name == "siqa":
        result = evaluate_siqa_core(evaluator, data_shard)
    elif task_name == "winogrande":
        result = evaluate_winogrande_core(evaluator, data_shard)
    elif task_name == "commonsenseqa":
        result = evaluate_commonsenseqa_core(evaluator, data_shard)
    elif task_name == "openbookqa":
        result = evaluate_openbookqa_core(evaluator, data_shard)
    elif task_name == "mmlu":
        result = evaluate_mmlu_core(evaluator, data_shard)
    else:
        result = {"correct": 0, "correct_norm": 0, "total": 0}
    
    print(f"[GPU {gpu_id}] Finished: {result.get('total', 0)} samples, acc={result.get('correct', 0)/max(result.get('total', 1), 1):.4f}")
    return result


def split_data_for_gpus(data: List[Dict], num_gpus: int) -> List[List[Dict]]:
    """Split data into roughly equal shards for each GPU"""
    shard_size = len(data) // num_gpus
    shards = []
    for i in range(num_gpus):
        start = i * shard_size
        end = start + shard_size if i < num_gpus - 1 else len(data)
        shards.append(data[start:end])
    return shards


def merge_results(shard_results: List[Dict]) -> Dict:
    """Merge results from multiple GPU shards"""
    total_correct = 0
    total_correct_norm = 0
    total_samples = 0
    subject_results = {}
    
    for result in shard_results:
        total_correct += result.get("correct", 0)
        total_correct_norm += result.get("correct_norm", 0)
        total_samples += result.get("total", 0)
        
        # Merge subject-wise results (for MMLU)
        for subj, subj_data in result.get("subject_results", {}).items():
            if subj not in subject_results:
                subject_results[subj] = {"correct": 0, "correct_norm": 0, "total": 0}
            subject_results[subj]["correct"] += subj_data.get("correct", 0)
            subject_results[subj]["correct_norm"] += subj_data.get("correct_norm", 0)
            subject_results[subj]["total"] += subj_data.get("total", 0)
    
    result = {
        "accuracy": total_correct / total_samples if total_samples > 0 else 0,
        "accuracy_norm": total_correct_norm / total_samples if total_samples > 0 else 0,
        "correct": total_correct,
        "correct_norm": total_correct_norm,
        "total": total_samples,
    }
    
    if subject_results:
        result["subject_results"] = {
            subj: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "accuracy_norm": data["correct_norm"] / data["total"] if data["total"] > 0 else 0,
                "total": data["total"],
            }
            for subj, data in subject_results.items()
        }
    
    return result


def evaluate_multi_gpu(task_name: str, data: List[Dict], gpu_ids: List[int],
                      checkpoint_path: str, model_type: str, hf_compatible: bool,
                      max_seq_len: int, batch_size: int, max_samples: int = None) -> Dict:
    """
    Evaluate a task using multiple GPUs in parallel.
    """
    if max_samples:
        data = data[:max_samples]
    
    num_gpus = len(gpu_ids)
    print(f"Running {task_name} on {num_gpus} GPUs: {gpu_ids}")
    print(f"Total samples: {len(data)}, ~{len(data)//num_gpus} per GPU")
    
    # Split data
    data_shards = split_data_for_gpus(data, num_gpus)
    
    # Create worker arguments
    worker_args = [
        (gpu_ids[i], data_shards[i], task_name, checkpoint_path, model_type, 
         hf_compatible, max_seq_len, batch_size)
        for i in range(num_gpus)
    ]
    
    # Use spawn method for CUDA compatibility
    ctx = mp.get_context('spawn')
    
    # Run parallel evaluation
    with ctx.Pool(num_gpus) as pool:
        shard_results = pool.starmap(evaluate_data_shard, worker_args)
    
    # Merge results
    return merge_results(shard_results)


# =============================================================================
# Task Evaluators (Core functions for multi-GPU)
# =============================================================================

def preprocess_hellaswag(text: str) -> str:
    """Preprocess HellaSwag text (from lighteval)"""
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def evaluate_hellaswag(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """Evaluate on HellaSwag with batch processing (supports both OpenCompass and lighteval formats)"""
    if max_samples:
        data = data[:max_samples]
    
    # Filter out samples without labels
    data = [item for item in data if item['label'] != -1]
    
    correct = 0
    correct_norm = 0
    total = 0
    
    # Process in batches
    sample_batch_size = evaluator.config.batch_size * 4  # 4 choices per HellaSwag
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            # Handle different data formats
            if item.get('format') == 'opencompass':
                ctx = item['query']
                endings = [" " + c for c in item['choices']]
            else:
                ctx = f"{item['ctx_a']} {item['ctx_b'].capitalize()} "
                ctx = preprocess_hellaswag(item['activity_label'] + ": " + ctx)
                endings = [" " + preprocess_hellaswag(e) for e in item['endings']]
            
            samples.append((ctx, endings))
            gold_indices.append(item['label'])
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            if pred_idx_norm == gold_idx:
                correct_norm += 1
            total += 1
        
        if total % 200 == 0 or batch_end == len(data):
            print(f"  HellaSwag: {total}/{len(data)}, acc={correct/total:.4f}, acc_norm={correct_norm/total:.4f}")
    
    return {
        'task': 'hellaswag',
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_norm': correct_norm / total if total > 0 else 0,
        'total': total,
    }


def evaluate_arc(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None, task_name: str = "arc") -> Dict:
    """Evaluate on ARC (Easy or Challenge) with batch processing"""
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    correct_norm = 0
    total = 0
    
    # Process in batches (ARC has variable number of choices, typically 4-5)
    sample_batch_size = evaluator.config.batch_size * 4
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            question = item['question']
            choices = [f" {c}" for c in item['choices']]
            samples.append((question, choices))
            gold_indices.append(item['gold_index'])
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            if pred_idx_norm == gold_idx:
                correct_norm += 1
            total += 1
        
        if total % 200 == 0 or batch_end == len(data):
            print(f"  {task_name}: {total}/{len(data)}, acc={correct/total:.4f}")
    
    return {
        'task': task_name,
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_norm': correct_norm / total if total > 0 else 0,
        'total': total,
    }


def evaluate_piqa(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """Evaluate on PiQA with batch processing"""
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    correct_norm = 0
    total = 0
    
    # Process in batches (PiQA has 2 choices)
    sample_batch_size = evaluator.config.batch_size * 8
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            question = item['goal']
            choices = [f" {item['sol1']}", f" {item['sol2']}"]
            samples.append((question, choices))
            gold_indices.append(item['label'])
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            if pred_idx_norm == gold_idx:
                correct_norm += 1
            total += 1
        
        if total % 500 == 0 or batch_end == len(data):
            print(f"  PiQA: {total}/{len(data)}, acc={correct/total:.4f}")
    
    return {
        'task': 'piqa',
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_norm': correct_norm / total if total > 0 else 0,
        'total': total,
    }


def evaluate_siqa(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """Evaluate on SocialIQA with batch processing"""
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    correct_norm = 0
    total = 0
    
    # Process in batches (SIQA has 3 choices)
    sample_batch_size = evaluator.config.batch_size * 5
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            context = item['context'] + " " + item['question']
            choices = [f" {item['answerA']}", f" {item['answerB']}", f" {item['answerC']}"]
            samples.append((context, choices))
            gold_indices.append(item['label'] - 1)  # 1-indexed to 0-indexed
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            if pred_idx_norm == gold_idx:
                correct_norm += 1
            total += 1
        
        if total % 500 == 0 or batch_end == len(data):
            print(f"  SIQA: {total}/{len(data)}, acc={correct/total:.4f}")
    
    return {
        'task': 'siqa',
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_norm': correct_norm / total if total > 0 else 0,
        'total': total,
    }


def evaluate_winogrande(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """Evaluate on Winogrande with batch processing"""
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    total = 0
    
    # Process in batches (Winogrande has 2 choices)
    sample_batch_size = evaluator.config.batch_size * 8
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            sentence = item['sentence']
            opt1 = sentence.replace("_", item['option1'])
            opt2 = sentence.replace("_", item['option2'])
            samples.append(("", [opt1, opt2]))
            gold_indices.append(int(item['answer']) - 1)  # 1-indexed to 0-indexed
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, _, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            total += 1
        
        if total % 500 == 0 or batch_end == len(data):
            print(f"  Winogrande: {total}/{len(data)}, acc={correct/total:.4f}")
    
    return {
        'task': 'winogrande',
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
    }


def evaluate_commonsenseqa(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """Evaluate on CommonsenseQA with batch processing"""
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    correct_norm = 0
    total = 0
    
    # Process in batches (CommonsenseQA has 5 choices)
    sample_batch_size = evaluator.config.batch_size * 3
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            question = item['question']
            choices = [f" {c}" for c in item['choices']]
            samples.append((question, choices))
            gold_indices.append(item['gold_index'])
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            if pred_idx_norm == gold_idx:
                correct_norm += 1
            total += 1
        
        if total % 500 == 0 or batch_end == len(data):
            print(f"  CommonsenseQA: {total}/{len(data)}, acc={correct/total:.4f}")
    
    return {
        'task': 'commonsenseqa',
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_norm': correct_norm / total if total > 0 else 0,
        'total': total,
    }


def evaluate_openbookqa(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """Evaluate on OpenBookQA with batch processing"""
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    correct_norm = 0
    total = 0
    
    # Process in batches (OpenBookQA has 4 choices)
    sample_batch_size = evaluator.config.batch_size * 4
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        
        for item in batch_data:
            question = item['question_stem']
            choices = [f" {c}" for c in item['choices']]
            samples.append((question, choices))
            gold_indices.append(item['gold_index'])
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            if pred_idx == gold_idx:
                correct += 1
            if pred_idx_norm == gold_idx:
                correct_norm += 1
            total += 1
        
        if total % 200 == 0 or batch_end == len(data):
            print(f"  OpenBookQA: {total}/{len(data)}, acc={correct/total:.4f}")
    
    return {
        'task': 'openbookqa',
        'accuracy': correct / total if total > 0 else 0,
        'accuracy_norm': correct_norm / total if total > 0 else 0,
        'total': total,
    }


def evaluate_mmlu(evaluator: NanoGPTEvaluator, data: List[Dict], max_samples: int = None) -> Dict:
    """
    Evaluate on MMLU with batch processing for better GPU utilization.
    
    This implementation matches the official lighteval_task.py prompt format:
    
        def mmlu_prompt(line, task_name: str = None):
            topic = line["subject"]
            prompt = f"The following are questions about {topic.replace('_', ' ')}.\nQuestion: "
            prompt += line["question"] + "\nAnswer:"
            
            return Doc(
                query=prompt,
                choices=[f" {c}" for c in line["choices"]],  # Note: space before each choice
                gold_index=line["answer"],
            )
    
    Metrics:
    - loglikelihood_acc: raw log-likelihood (no normalization)
    - loglikelihood_acc_norm_nospace: length-normalized log-likelihood
    """
    if max_samples:
        data = data[:max_samples]
    
    correct = 0
    correct_norm = 0
    total = 0
    subject_results = {}
    
    # Prepare all samples for batch processing
    # Process in chunks of samples (each sample has 4 choices)
    sample_batch_size = evaluator.config.batch_size * 4  # 4 choices per MMLU question
    
    for batch_start in range(0, len(data), sample_batch_size):
        batch_end = min(batch_start + sample_batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        samples = []
        gold_indices = []
        subjects = []
        
        for item in batch_data:
            subject = item.get('subject', 'unknown')
            subjects.append(subject)
            
            # Exact prompt format from lighteval_task.py
            topic = subject.replace('_', ' ')
            prompt = f"The following are questions about {topic}.\nQuestion: "
            prompt += item['question'] + "\nAnswer:"
            
            # Choices with leading space (exactly as in lighteval_task.py)
            choices = [f" {c}" for c in item['choices']]
            
            gold_idx = item.get('answer', item.get('gold_index', 0))
            gold_indices.append(gold_idx)
            
            samples.append((prompt, choices))
        
        # Batch evaluation
        batch_results = evaluator.score_batch(samples)
        
        # Process results
        for i, (pred_idx, pred_idx_norm, _, _) in enumerate(batch_results):
            gold_idx = gold_indices[i]
            subject = subjects[i]
            
            is_correct = pred_idx == gold_idx
            is_correct_norm = pred_idx_norm == gold_idx
            
            if is_correct:
                correct += 1
            if is_correct_norm:
                correct_norm += 1
            total += 1
            
            # Track per-subject results
            if subject not in subject_results:
                subject_results[subject] = {'correct': 0, 'correct_norm': 0, 'total': 0}
            subject_results[subject]['total'] += 1
            if is_correct:
                subject_results[subject]['correct'] += 1
            if is_correct_norm:
                subject_results[subject]['correct_norm'] += 1
        
        # Progress logging
        if total % 500 == 0 or batch_end == len(data):
            print(f"  MMLU: {total}/{len(data)}, acc={correct/total:.4f}, acc_norm={correct_norm/total:.4f}")
    
    return {
        'task': 'mmlu',
        'accuracy': correct / total if total > 0 else 0,  # loglikelihood_acc
        'accuracy_norm': correct_norm / total if total > 0 else 0,  # loglikelihood_acc_norm_nospace
        'correct': correct,
        'correct_norm': correct_norm,
        'total': total,
        'subject_results': subject_results,
    }


# =============================================================================
# Core evaluation functions (for multi-GPU support)
# These return raw counts for merging across GPUs
# =============================================================================

def evaluate_hellaswag_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core HellaSwag evaluation - returns raw counts"""
    result = evaluate_hellaswag(evaluator, data, max_samples=None)
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result.get('accuracy_norm', result['accuracy']) * result['total']),
        "total": result['total'],
    }

def evaluate_arc_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core ARC evaluation - returns raw counts"""
    result = evaluate_arc(evaluator, data, max_samples=None, task_name="arc")
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result.get('accuracy_norm', result['accuracy']) * result['total']),
        "total": result['total'],
    }

def evaluate_piqa_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core PiQA evaluation - returns raw counts"""
    result = evaluate_piqa(evaluator, data, max_samples=None)
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result.get('accuracy_norm', result['accuracy']) * result['total']),
        "total": result['total'],
    }

def evaluate_siqa_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core SIQA evaluation - returns raw counts"""
    result = evaluate_siqa(evaluator, data, max_samples=None)
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result.get('accuracy_norm', result['accuracy']) * result['total']),
        "total": result['total'],
    }

def evaluate_winogrande_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core Winogrande evaluation - returns raw counts"""
    result = evaluate_winogrande(evaluator, data, max_samples=None)
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result['accuracy'] * result['total']),  # No norm for winogrande
        "total": result['total'],
    }

def evaluate_commonsenseqa_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core CommonsenseQA evaluation - returns raw counts"""
    result = evaluate_commonsenseqa(evaluator, data, max_samples=None)
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result.get('accuracy_norm', result['accuracy']) * result['total']),
        "total": result['total'],
    }

def evaluate_openbookqa_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core OpenBookQA evaluation - returns raw counts"""
    result = evaluate_openbookqa(evaluator, data, max_samples=None)
    return {
        "correct": round(result['accuracy'] * result['total']),
        "correct_norm": round(result.get('accuracy_norm', result['accuracy']) * result['total']),
        "total": result['total'],
    }

def evaluate_mmlu_core(evaluator: NanoGPTEvaluator, data: List[Dict]) -> Dict:
    """Core MMLU evaluation - returns raw counts with subject breakdown"""
    result = evaluate_mmlu(evaluator, data, max_samples=None)
    return {
        "correct": result.get('correct', round(result['accuracy'] * result['total'])),
        "correct_norm": result.get('correct_norm', round(result.get('accuracy_norm', result['accuracy']) * result['total'])),
        "total": result['total'],
        "subject_results": result.get('subject_results', {}),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Offline LightEval evaluation for NanoGPT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--model_type", type=str, default="gpt2-xl", 
                        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                        help="Model architecture type")
    parser.add_argument("--hf_compatible", action="store_true", 
                        help="Use HF-compatible architecture (LayerNorm, learned pos emb)")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for eval")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for inference (increase for better GPU utilization, e.g., 128-256 for H200)")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Max samples per task (for faster debugging)")
    parser.add_argument("--output_dir", type=str, default="lighteval_results", help="Output directory")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated tasks or 'all'. Available: hellaswag,winogrande,piqa,siqa,arc_easy,arc_challenge,commonsenseqa,openbookqa,mmlu")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory (default: opencompass/data)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for parallel evaluation")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). If not set, uses 0 to num_gpus-1")
    
    args = parser.parse_args()
    
    # Setup data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = PROJECT_ROOT / "opencompass" / "data"
    
    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))
    
    use_multi_gpu = args.num_gpus > 1
    
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"GPUs: {gpu_ids} ({'multi-GPU' if use_multi_gpu else 'single-GPU'} mode)")
    
    # For single GPU, create evaluator upfront
    evaluator = None
    if not use_multi_gpu:
        config = EvalConfig(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            hf_compatible=args.hf_compatible,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
        evaluator = NanoGPTEvaluator(config)
    
    # Parse tasks
    if args.tasks == "all":
        tasks = ["hellaswag", "winogrande", "piqa", "siqa", "arc_easy", "arc_challenge", 
                 "commonsenseqa", "openbookqa", "mmlu"]
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
    
    # Run evaluations
    results = {}
    
    # Helper function for multi-GPU evaluation
    def run_eval(task_name: str, data: List[Dict], eval_func, **kwargs):
        if use_multi_gpu:
            return evaluate_multi_gpu(
                task_name=task_name,
                data=data,
                gpu_ids=gpu_ids,
                checkpoint_path=args.checkpoint,
                model_type=args.model_type,
                hf_compatible=args.hf_compatible,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
        else:
            return eval_func(evaluator, data, args.max_samples, **kwargs)
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating: {task}")
        print(f"{'='*60}")
        
        if task == "hellaswag":
            data = load_hellaswag_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_hellaswag)
                results[task] = result
        
        elif task == "arc_easy":
            data = load_arc_offline(data_dir, "ARC-e")
            if data:
                result = run_eval(task, data, evaluate_arc, task_name="arc_easy")
                results[task] = result
        
        elif task == "arc_challenge":
            data = load_arc_offline(data_dir, "ARC-c")
            if data:
                result = run_eval(task, data, evaluate_arc, task_name="arc_challenge")
                results[task] = result
        
        elif task == "piqa":
            data = load_piqa_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_piqa)
                results[task] = result
        
        elif task == "siqa":
            data = load_siqa_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_siqa)
                results[task] = result
        
        elif task == "winogrande":
            data = load_winogrande_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_winogrande)
                results[task] = result
        
        elif task == "commonsenseqa":
            data = load_commonsenseqa_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_commonsenseqa)
                results[task] = result
        
        elif task == "openbookqa":
            data = load_openbookqa_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_openbookqa)
                results[task] = result
        
        elif task == "mmlu":
            data = load_mmlu_offline(data_dir)
            if data:
                result = run_eval(task, data, evaluate_mmlu)
                results[task] = result
        
        else:
            print(f"Unknown task: {task}")
            continue
        
        if task in results:
            print(f"\n{task} Results:")
            print(f"  Accuracy: {results[task]['accuracy']*100:.2f}%")
            if 'accuracy_norm' in results[task]:
                print(f"  Accuracy (norm): {results[task]['accuracy_norm']*100:.2f}%")
            print(f"  Total samples: {results[task]['total']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_acc = []
    total_acc_norm = []
    
    for task, result in results.items():
        acc = result.get('accuracy', 0) * 100
        acc_norm = result.get('accuracy_norm', acc) * 100
        total_acc.append(acc)
        total_acc_norm.append(acc_norm)
        print(f"{task:20s}: {acc:6.2f}% (norm: {acc_norm:6.2f}%)")
    
    if total_acc:
        print(f"\nAverage accuracy: {np.mean(total_acc):.2f}%")
        print(f"Average accuracy (norm): {np.mean(total_acc_norm):.2f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract checkpoint name for output file
    ckpt_name = Path(args.checkpoint).stem
    output_file = output_dir / f"results_{ckpt_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'model_type': args.model_type,
            'results': results,
            'average_accuracy': np.mean(total_acc) if total_acc else 0,
            'average_accuracy_norm': np.mean(total_acc_norm) if total_acc_norm else 0,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

