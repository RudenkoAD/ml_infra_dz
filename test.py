import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_llm():
    """Test the LLM model by generating a simple response."""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("nilq/mistral-1L-tiny")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained("nilq/mistral-1L-tiny").to(device)
        model.generation_config.pad_token_id = tokenizer.eos_token_id

        # Test input
        test_prompt = "Hello, how are you? test ducking"
        inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

        # Generate response
        outputs = model.generate(inputs, max_length=50, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Test Prompt:", test_prompt)
        print("Generated Response:", response[len(test_prompt):].strip())
    except Exception as e:
        print("Error during LLM testing:", str(e))

if __name__ == "__main__":
    test_llm()
