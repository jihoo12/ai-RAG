import os
import re
import torch
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. LOAD MODELS
# ==========================================
model_name = "google/gemma-4-E2B-it"
print(f"Loading {model_name} and Embedding models...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ==========================================
# 2. RAG FUNCTIONS (Updated clean_latex)
# ==========================================
def clean_latex(text):
    """
    Cleans LaTeX source and translates custom macros to standard symbols 
    to ensure the Markdown previewer can render the AI's output.
    """
    # 1. Remove LaTeX comments (lines starting with %)
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.M)
    
    # 2. Remove \label{...} tags
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    
    # 3. Define a translation map for your custom thesis macros
    # Add any other custom commands from your thesis here!
    translations = {
        r"\der": r"\vdash",
        r"\U": r"\mathcal{U}",
        r"\Glue": r"\mathsf{Glue}",
        r"\Equiv": r"\mathsf{Equiv}",
        r"\Path": r"\mathsf{Path}",
        r"\can": r"\mathsf{can}",
        r"\myeq{}": r"\simeq",
        r"\myeq": r"\simeq",
    }
    
    # Apply translations
    for cmd, replacement in translations.items():
        # We use a word boundary \b to avoid replacing \universe if we just want \U
        # However, for LaTeX commands, a simple replace is often safer
        text = text.replace(cmd, replacement)
        
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
        
    print(f"Processing .tex files in {directory}...")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    # Apply cleaning here during the load phase
                    content = clean_latex(f.read())
                    file_chunks = splitter.split_text(content)
                    chunks.extend([f"From file {file}:\n{c}" for c in file_chunks])
    return chunks

# Pre-load and Index the data
raw_chunks = load_tex_files("./RAG-data")
if raw_chunks:
    embeddings = embed_model.encode(raw_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
else:
    index = None
    print("No .tex files found. RAG will not have context.")

def retrieve_context(query, k=3):
    if not raw_chunks or index is None: return ""
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec).astype('float32'), k)
    return "\n---\n".join([raw_chunks[i] for i in indices[0]])

# ==========================================
# 3. INTERACTIVE CHAT FUNCTION
# ==========================================
def chat_with_rag():
    model.eval()
    
    history = [
        {
            "role": "user", 
            "content": (
                "You are a technical assistant specializing in Homotopy Type Theory. "
                "Rules for your responses:\n"
                "1. Use standard LaTeX symbols only. Avoid custom thesis macros.\n"
                "2. Ensure symbols like \\vdash, \\mathcal{U}, and \\mathsf{Glue} are used.\n"
                "3. Use inline $...$ and block $$...$$ for all math.\n"
                "4. Be conversational but mathematically rigorous."
            )
        },
        {
            "role": "assistant",
            "content": "I am ready. I will provide rigorous HoTT explanations using standard LaTeX notation for your previewer."
        }
    ]
    
    print("\n" + "="*50)
    print("HoTT RAG CHAT SESSION START")
    print("Type 'quit' to exit.")
    print("="*50)

    while True:
        try:
            user_input = input("\n[User]: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                break
            if not user_input:
                continue

            context_str = retrieve_context(user_input)
            
            # Send context to the model for this turn
            prompt_with_context = (
                f"CONTEXT FROM THESIS:\n{context_str}\n\n"
                f"USER QUESTION: {user_input}"
            )
            
            current_messages = history + [{"role": "user", "content": prompt_with_context}]
            prompt = tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            print(f"[A]: ", end="")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    streamer=streamer,
                    do_sample=True,
                    temperature=0.3,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Cleanly update history without the bulky context
            new_tokens = output_ids[0][inputs.input_ids.shape[-1]:]
            assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": assistant_response})
            
            if len(history) > 10:
                history = [history[0], history[1]] + history[-8:]

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    try:
        chat_with_rag()
    except Exception as e:
        print(f"\nCritical Error: {e}")