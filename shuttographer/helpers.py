from transformers import pipeline
import torch
import requests
import json
import os
import base64

api_key = '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'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

device="cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

def speech_to_text(audio_file):
    return pipe(audio_file)

def call_chatbot_api(text):
    prompt = f"I have just asked someone if I can take their photograph. They replied: {text}. Did they consent to have their photograph taken? It is of vital importance that you respond to this question with only one word, either 'yes' or 'no', formatted exactly as I have written it here (no punctuation and lower case)."
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.3
    }
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response_dict = json.loads(response.text)
            llm_response = response_dict['choices'][0]['message']['content']
            print('llm_response')
            return llm_response
        except:
            print(f"error at {N}, retrying")
    llm_response = "API ERROR"
    return llm_response

def stable_diffusion_edit(prompt, image_path):
    response = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer sk-6sn8P2lbd2nIIp6QPO5LSgtU49kA0vlNGi49uRODgQLAgP92"
        },
        files={
            "init_image": open(image_path, "rb")
        },
        data={
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.4,
            "steps": 40,
            "seed": 0,
            "cfg_scale": 10,
            "samples": 1,
            "style_preset": "fantasy-art",
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": 1,
            "text_prompts[1][text]": 'blurry, bad',
            "text_prompts[1][weight]": -1,
        }
    )
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    if not os.path.exists("./out"):
        os.makedirs("./out")

    for i, image in enumerate(data["artifacts"]):
        with open(f'./out/img2img_0.png', "wb") as f:
            f.write(base64.b64decode(image["base64"]))


def stable_diffusion_upscale(image_path):
    response = requests.post(
        "https://api.stability.ai/v1/generation/esrgan-v1-x2plus/image-to-image/upscale",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer sk-6sn8P2lbd2nIIp6QPO5LSgtU49kA0vlNGi49uRODgQLAgP92"
        },
        files={
            "image": open(f'/content/out/img2img_0.png', "rb")
        },
        data={
            "height": 1024
        }
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    # make sure the out directory exists
    if not os.path.exists("./out"):
        os.makedirs("./out")

    for i, image in enumerate(data["artifacts"]):
        with open(f'./out/upscale_{image["seed"]}.png', "wb") as f:
            f.write(base64.b64decode(image["base64"]))

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    # make sure the out directory exists
    if not os.path.exists("./out"):
        os.makedirs("./out")

    for i, image in enumerate(data["artifacts"]):
        with open(f'./out/img2img_0.png', "wb") as f:
            f.write(base64.b64decode(image["base64"]))