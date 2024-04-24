import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint_path = "Training/results/gemma-oxford/checkpoint-500"

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

rouge_metric = load_metric("rouge")
bertscore_metric = load_metric("bertscore")

def generate_definition(model, tokenizer, word, pos, max_length=150, top_p=0.9, temperature=0.9, num_beams=3, do_sample=True):
    prompt = f"[BOS] {word} (POS: {pos}) <definition>"

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=do_sample,
        top_k=50,
        top_p=top_p,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    definition = tokenizer.decode(output[0], skip_special_tokens=False)
    definition = definition.split('<definition>')[1] if '<definition>' in definition else definition

    if ' [' in definition:
        main_definition = definition.split(' [')[0].strip()
        etymology = '[' + definition.split(' [', 1)[1].strip()
        etymology = etymology.split(']')[0] + ']' if ']' in etymology else etymology
    else:
        main_definition = definition
        etymology = ""

    clean_definition = f"{main_definition} {etymology}".replace("<bos>", "").strip()

    return clean_definition

csv_file_path = "Evaluation/evaluation_results.csv"

with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    if file.tell() == 0:
        csv_writer.writerow(["Word", "POS", "Generated Definition", "Expected Definition", "ROUGE-L F1", "BERTScore F1"])
    
    try:
        while True:
            word = input("Enter the word (or 'quit' to stop): ")
            if word.lower() == 'quit':
                break
            pos = input("Enter the part of speech: ")

            generated_definition = generate_definition(model, tokenizer, word, pos)
            print(f"Generated Definition: {generated_definition}\n")

            expected_definition = input("Enter the expected definition for metric calculation: ")

            rouge_score = rouge_metric.compute(predictions=[generated_definition], references=[expected_definition])
            bertscore_result = bertscore_metric.compute(predictions=[generated_definition], references=[expected_definition], lang="en")

            print(f"ROUGE-L F1: {rouge_score['rougeL'].mid.fmeasure}")
            print(f"BERTScore F1: {bertscore_result['f1'][0]}")
            print("\n---\n")

            csv_writer.writerow([
                word,
                pos,
                generated_definition,
                expected_definition,
                rouge_score['rougeL'].mid.fmeasure,
                bertscore_result['f1'][0]
            ])
            file.flush()
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Results saved to {csv_file_path}")
