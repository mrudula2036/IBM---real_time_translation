from transformers import MarianMTModel, MarianTokenizer
import torch

# Step 1: Load a pre-trained NMT model and tokenizer
def load_model_and_tokenizer():
    model_name = "Helsinki-NLP/opus-mt-en-hi"  # English to Hindi model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

# Step 2: Translate text using the NMT model
def translate_text(model, tokenizer, text, device):
    # Tokenize the input text
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Move tensors to the same device as model
    tokenized_text = {key: val.to(device) for key, val in tokenized_text.items()}

    # Perform translation
    translated_tokens = model.generate(**tokenized_text)

    # Decode the translated tokens back into text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Step 3: Real-time translation loop
def real_time_translation():
    model, tokenizer, device = load_model_and_tokenizer()
    print("Real-Time Language Translation System (English to Hindi)")
    print("Type 'exit' to quit.")

    while True:
        # Get user input
        text = input("Enter text in English: ")
        if text.lower() == "exit":
            print("Exiting the translation system. Goodbye!")
            break
        # Translate the input text
        translated_text = translate_text(model, tokenizer, text, device)
        print(f"Translated text in Hindi: {translated_text}\n")

# Run the real-time translation system
if __name__ == "__main__":
    real_time_translation()
