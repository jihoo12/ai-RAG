import os
import re
import torch
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. LOAD MODELS (Moved to top)
model_name = "google/gemma-4-E2B-it"
print("Loading Gemma and Embedding models...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 2. RAG FUNCTIONS
def clean_latex(text):
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.M)
    return text

def load_tex_files(directory):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=150,
        separators=["\n\\section", "\n\\subsection", "\n\\begin", "\n\n", "\n", " "]
    )
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found!")
        return []
        
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = clean_latex(f.read())
                    file_chunks = splitter.split_text(content)
                    chunks.extend([f"From file {file}:\n{c}" for c in file_chunks])
    return chunks

# Pre-load the data
raw_chunks = load_tex_files("./RAG-data")
if raw_chunks:
    embeddings = embed_model.encode(raw_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
else:
    print("No .tex files found. RAG will not have context.")

def retrieve_context(query, k=3):
    if not raw_chunks: return ""
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k)
    return "\n---\n".join([raw_chunks[i] for i in indices[0]])

# 3. GENERATION FUNCTION
def generate_smart_text_with_rag(user_prompt):
    model.eval()
    context_str = retrieve_context(user_prompt)
    
    # FORMAT FOR GEMMA CHAT
    full_instruction = f"""You are an expert in Homotopy Type Theory. 
Using ONLY the LaTeX fragments provided below, provide a detailed and 
technical explanation to the question. Use LaTeX for all math.
CONTEXT:
{context_str}

QUESTION:
{user_prompt}"""

    messages = [{"role": "user", "content": full_instruction}]
    
    # APPLY CHAT TEMPLATE (Important for Gemma)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print(f"\n[A]: ", end="")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1024,
            streamer=streamer,
            do_sample=True,
            temperature=0.3, # Lowered for better technical accuracy
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

if __name__ == "__main__":
    test_prompt = "What does the thesis say about Higher-Dimensional Types?"
    try:
        generate_smart_text_with_rag(test_prompt)
    except Exception as e:
        print(f"\nException: {e}")