from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

app = Flask(__name__, static_folder='static')

def generate_text(prompt, max_length):
    generator = pipeline('text-generation', model='gpt2')
    generated_text = generator(prompt, max_length=max_length, num_return_sequences=1, truncation=True)
    return generated_text[0]['generated_text']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        max_length = int(request.form['max_length'])
        generated_text = generate_text(prompt, max_length)
        return render_template('index.html', prompt=prompt, generated_text=generated_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
