import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_multiplication_problems(n_digits):
    max_num = 10**n_digits - 1
    a = np.random.randint(1, max_num + 1)
    b = np.random.randint(1, max_num + 1)
    return a, b, a * b

def evaluate_model(model, tokenizer, n_digits, num_samples=100):
    correct = 0
    for _ in tqdm(range(num_samples)):
        a, b, result = generate_multiplication_problems(n_digits)
        prompt = f"What is {a} multiplied by {b}? Answer with just the number."
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            predicted = int(response.split()[-1])
            if predicted == result:
                correct += 1
        except ValueError:
            pass
    
    return correct / num_samples

def main():
    model_name = "decapoda-research/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    max_digits = 3
    accuracies = np.zeros((max_digits, max_digits))

    for i in range(1, max_digits + 1):
        for j in range(1, max_digits + 1):
            print(f"Evaluating {i} digit * {j} digit multiplication")
            accuracy = evaluate_model(model, tokenizer, max(i, j))
            accuracies[i-1, j-1] = accuracy

    plt.figure(figsize=(10, 8))
    plt.imshow(accuracies, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Accuracy')
    plt.title('LLaMA 7B Multiplication Accuracy')
    plt.xlabel('Number of digits (second operand)')
    plt.ylabel('Number of digits (first operand)')
    plt.xticks(range(max_digits), range(1, max_digits + 1))
    plt.yticks(range(max_digits), range(1, max_digits + 1))

    for i in range(max_digits):
        for j in range(max_digits):
            plt.text(j, i, f'{accuracies[i, j]:.2f}', ha='center', va='center', color='white')

    plt.savefig('llama_multiplication_accuracy.png')
    plt.show()

if __name__ == "__main__":
    main()
