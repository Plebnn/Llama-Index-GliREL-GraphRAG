import os
import json
import logging
from typing import Dict, List
from dotenv import load_dotenv
from transformers import AutoTokenizer
from tqdm import tqdm
import asyncio

#llama index imports
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex,Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext, load_index_from_storage
import openlit

# Load environment variables
load_dotenv()

openlit.init(
  otlp_endpoint="http://127.0.0.1:4318",
  application_name="query",
  environment="obama_enviroment")
  


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llamaindex_processing.log")
    ]
)
# initailize Llama index Ollama connection
llm = Ollama(
    model= "gemma3:12b",
    request_timeout=120.0,
    context_window=8128,
    temperature=0.0
)

Settings.llm = llm
Settings.chunk_size=512
Settings.chunk_overlap=64

embed_model = OllamaEmbedding(
    model_name="snowflake-arctic-embed2:latest",
    ollama_additional_kwargs={"mirostat": 0},
)
Settings.embed_model = embed_model

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

def split_text(
    text: str, 
    tokenizer: AutoTokenizer, 
    chunk_token_size: int = 256, 
    chunk_overlap_token_size: int = 32
) -> List[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks

async def process_corpus(
    corpus_name: str,
    questions: List[dict],
    mode: str
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Prepare output directory
    output_dir = f"./.persistent_storage/.results/{mode}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # initialize Llama-index retrieval engine
    strorage_cotext = StorageContext.from_defaults(persist_dir=f"./.persistent_storage/{mode}/{corpus_name}")


    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare queries and gold answers
    all_queries = [q["question"] for q in corpus_questions]
    gold_answers = [[q['answer']] for q in corpus_questions]
    
    # initlaize RAG engine
    index = load_index_from_storage(storage_context=strorage_cotext)


    logging.info(f"✅ Indexed corpus: {corpus_name}")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="compact",
        similarity_top_k=8,
        embedding_mode="hybrid",
        include_text=True, 
    )

    # Process questions
    results = []
    solutions =[]
    for query in all_queries:
        #nest_asyncio.apply()
        response_object = await query_engine.aquery(query)
        solution_dict = {"question":query,
                         "answer":response_object.response,
                         "docs":response_object.get_formatted_sources(10000)
                         }
        solutions.append(solution_dict)
    for question in corpus_questions:
        solution = next((sol for sol in solutions if sol['question'] == question['question']), None)
        if solution:
            results.append({
                "id": question["id"],
                "question": question["question"],
                "source": corpus_name,
                "context": solution.get("docs", ""),
                "evidence": question.get("evidence", ""),
                "question_type": question.get("question_type", ""),
                "generated_answer": solution.get("answer", ""),
                "gold_answer": question.get("answer", "")
            })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")
    #print(results)

def main():

    # Load corpus data
    try:
        with open("../.data/novel.json", "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from ../.data/novel.json")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    
    # Load question data
    try:
        with open("../.data/novel_questions.json", "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from ../.data/novel_questions.json")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return
    
    # Process each corpus in the subset
    for mode in ["gli","hybrid","llm"]:
        for item in corpus_data:
            corpus_name = item["corpus_name"]
            context = item["context"]
            process_corpus(
                corpus_name=corpus_name,
                questions=grouped_questions,
                mode=mode
            )

if __name__ == "__main__":
    main()