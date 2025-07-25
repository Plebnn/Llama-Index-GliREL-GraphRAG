import asyncio
import argparse
import json
import numpy as np
import os
from typing import Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from Evaluation.metrics import compute_answer_correctness, compute_coverage_score, compute_faithfulness_score, compute_rouge_score
from pathlib import Path

async def evaluate_dataset(
    dataset: Dataset,
    metrics: List[str],
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate the metric scores on the entire dataset."""
    results = {metric: [] for metric in metrics}
    
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    # Create a list of tasks
    tasks = []
    for i in range(total_samples):
        tasks.append(
            evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings
            )
        )
    
    # Collect results and display progress
    sample_results = []
    for i, future in enumerate(asyncio.as_completed(tasks)):
        result = await future
        sample_results.append(result)
        print(f"Completed sample {i+1}/{total_samples} - {((i+1)/total_samples)*100:.1f}%")
    
    # Aggregate results
    for sample in sample_results:
        for metric, score in sample.items():
            if isinstance(score, (int, float)) and not np.isnan(score):
                results[metric].append(score)
    
    return {metric: np.nanmean(scores) for metric, scores in results.items()}

async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    metrics: List[str],
    llm: BaseLanguageModel,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate the metric scores for a single sample."""
    results = {}
    
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )
    
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, ground_truth, answer, llm
        )
    
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )
    
    task_results = await asyncio.gather(*tasks.values())
    
    for i, metric in enumerate(tasks.keys()):
        results[metric] = task_results[i]
    
    return results

async def main(model,base_url,bge_model,data_file,output_file):
    """Main evaluation function that accepts command-line arguments."""
    # Check if the API key is set
    if not os.getenv("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY environment variable is not set")
    
    # Initialize the model
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0.0,
        max_retries=3,
        timeout=60
    )
    
    # Initialize the embedding model
    embedding = HuggingFaceBgeEmbeddings(model_name=bge_model)
    
    # Load evaluation data
    print(f"Loading evaluation data from {data_file}...")
    with open(data_file, 'r') as f:
        file_data = json.load(f)  # Now a list of question items
    
    # Define the evaluation metrics for each question type
    metric_config = {
        'Fact Retrieval': ["rouge_score", "answer_correctness"],
        'Complex Reasoning': ["rouge_score", "answer_correctness"],
        'Contextual Summarize': ["answer_correctness", "coverage_score"],
        'Creative Generation': ["answer_correctness", "coverage_score", "faithfulness"]
    }
    
    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    
    all_results = {}
    
    # Evaluate each found question type (only those in metric_config)
    for question_type in list(grouped_data.keys()):
        # Skip types not defined in metric_config
        if question_type not in metric_config:
            print(f"Skipping undefined question type: {question_type}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]
        questions = [item['question'] for item in group_items]
        ground_truths = [item['gold_answer'] for item in group_items]
        answers = [item['generated_answer'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)
        
        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            metrics=metric_config[question_type],
            llm=llm, 
            embeddings=embedding  
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
    
    # Save final results
    if output_file:
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\nFinal Evaluation Summary:")
    print("=" * 50)
    for q_type, metrics in all_results.items():
        print(f"\nQuestion Type: {q_type}")
        for metric, score in metrics.items():
            print(f"  {metric}: {score:.4f}")
    
    print('\nEvaluation complete.')

if __name__ == "__main__":
    with open('./Datasets/Corpus/novel.json', 'r') as file:
    # Load the JSON data from the file into a Python object
        data = json.load(file)

    modes = [
        "embed",#"gli","hybrid","llm"
        ]
    # main parameters
    model= "gpt-4.1-nano"
    base_url = "https://api.openai.com/v1"
    bge_model = "BAAI/bge-large-en-v1.5"

    for mode in modes:
        for novel in data:
            corpus_name = novel["corpus_name"]
            data_file = f"/home/nick/projects/Llama-Index-GliREL-GraphRAG/evaluation/.persistent_storage/.results2/{mode}/{corpus_name}/predictions_{corpus_name}.json"
            output_file = f"./results/{mode}/{corpus_name}_evaluation_results.json"
            if Path.exists(Path(output_file)):
                print(f"eval for {mode}: {corpus_name} allready exists at {output_file}")
                continue

            asyncio.run(main(model=model,base_url=base_url,bge_model=bge_model,data_file=data_file,output_file=output_file))

    
    