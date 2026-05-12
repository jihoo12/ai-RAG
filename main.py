import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_name = "google/gemma-4-E2B-it"

print("loading please wait...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"  
)

def generate_smart_text(user_prompt):
    model.eval()
    
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print(f"\n[Q]: {user_prompt}")
    print("-" * 30)
    print("[A]: ", end="")
    
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1024,      
            streamer=streamer,        
            do_sample=True,           
            temperature=0.7,          
            top_p=0.9,                
            repetition_penalty=1.2,   
            pad_token_id=tokenizer.eos_token_id
        )
    print("\n" + "-" * 30)

if __name__ == "__main__":
    test_prompt = "give me haskell example code"
    
    try:
        generate_smart_text(test_prompt)
    except KeyboardInterrupt:
        print("\nterminated by user keyboard interrupt")
    except Exception as e:
        print(f"\nexception: {e}")