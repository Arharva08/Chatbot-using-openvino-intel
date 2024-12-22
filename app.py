from flask import Flask, request, jsonify
import numpy as np
import openvino as ov
from simple_tokenizer import SimpleTokenizer
from misc import sampling

app = Flask(__name__)

model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model_vendor, model_name = model_id.split('/')
model_precision = 'INT4'

tokenizer = SimpleTokenizer(model_vendor, model_name)

device = 'CPU'
ov_core = ov.Core()
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache"}
ov_model = ov_core.read_model(model=f'{model_name}/{model_precision}/openvino_model.xml')
compiled_model = ov_core.compile_model(ov_model, device, ov_config)
infer_request = compiled_model.create_infer_request()

def generate_response(question):
    prompt_text = f"""\
    You are a helpful, respectful and honest medical assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.</s>
    {question}</s>
    """
    input_tokens = tokenizer(text=prompt_text, return_tensors='np')
    input_ids = input_tokens.input_ids
    attention_mask = input_tokens.attention_mask
    position = input_ids.shape[-1]
    position_ids = np.array([range(position)], dtype=np.int64)
    beam_idx = np.array([0], dtype=np.int32)

    temperature = 0.7  # Lower temperature to make the model less random
    top_p = 0.9  # Increase top_p to consider more probable tokens
    top_k = 50  # Increase top_k to consider a larger set of tokens
    eos_token_id = tokenizer.eos_token_id
    num_max_token_for_generation = 300

    infer_request.reset_state()
    generated_text_ids = np.array([], dtype=np.int32)

    for _ in range(num_max_token_for_generation):
        response = infer_request.infer(inputs={'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids, 'beam_idx': beam_idx})
        logits = response['logits'][0, -1, :]
        sampled_id = sampling(logits, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=True)

        if sampled_id == eos_token_id:
            break

        generated_text_ids = np.append(generated_text_ids, sampled_id)
        input_ids = np.array([[sampled_id]], dtype=np.int64)
        attention_mask = np.array([[1]], dtype=np.int64)
        position_ids = np.array([[position]], dtype=np.int64)
        beam_idx = np.array([0], dtype=np.int32)
        position += 1

    output_text = tokenizer.decode(generated_text_ids)
    return output_text

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data['question']
    response = generate_response(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
