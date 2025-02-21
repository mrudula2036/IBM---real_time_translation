from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import torch

app = Flask(__name__)

# Load Model and Tokenizer
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate_text(text):
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokenized_text = {key: val.to(device) for key, val in tokenized_text.items()}
    translated_tokens = model.generate(**tokenized_text)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    translated_text = translate_text(text)
    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run(debug=True)
