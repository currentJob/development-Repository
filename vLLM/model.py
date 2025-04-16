import os
import re
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

def generate_quiz_from_subject(prompt: str) -> str:
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "/model",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AI specialized in creating quiz questions. "
                    "When the user provides a subject, generate one high-quality multiple-choice question related to that subject. "
                    "Provide exactly five answer choices labeled A to E, and clearly indicate the correct answer at the end. "
                    "Also, provide a brief explanation of why the correct answer is right. "
                    "The question should be appropriate for general learners. "
                    "Format your response as follows:\n\n"
                    "Subject: [subject]\n"
                    "Question: [your question]\n"
                    "A. [option A]\n"
                    "B. [option B]\n"
                    "C. [option C]\n"
                    "D. [option D]\n"
                    "E. [option E]\n"
                    "Answer: [Correct option letter]\n"
                    "Explanation: [Why the answer is correct]"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.split("</think>")[-1].strip()
    except Exception as e:
        print("vLLM 요청 실패:", e)
        return ""

def load_translation_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def translate_text(model, tokenizer, text: str) -> str:
    prompt = f"""
    ### Instruction:
    주어진 텍스트를 한국어로 번역하세요.
    ### Input:
    '''{text}'''
    ### Output:
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        temperature=0.1,
        top_p=1,
        repetition_penalty=1.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("Output:")[-1].strip()

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\\n', '\n')
    text = '\n'.join(line for line in text.splitlines() if 'table' not in line.lower())
    text = text.replace('*', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s?([A-E])\.', r'\n\1.', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def generate_translated_quiz(subject: str) -> str:
    raw_response = generate_quiz_from_subject(subject)
    if not raw_response:
        return "퀴즈 생성 실패"
    
    translator_model_path = "./models/Llama-3.2-3B-KO-EN-Translation"

    model, tokenizer = load_translation_model(translator_model_path)
    translated = translate_text(model, tokenizer, raw_response)
    return clean_text(translated)

if __name__ == "__main__":
    quiz = generate_translated_quiz("Computer Science")

    print("\n======= 퀴즈 =======\n")
    print(quiz)
