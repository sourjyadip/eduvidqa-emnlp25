import base64
import requests
import google.generativeai as genai
import json

with open('config.json', 'r') as f:
    config = json.load(f)

def gpt4_response(prompt):
    # openAI API Key
    openai_api_key = config["LLM model"]["openai_api_key"]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt,
            }
        ]
        }
    ],
    "max_tokens": 500
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    #print(response.json())
    response_json  = response.json()
    content = str(response_json['choices'][0]['message']['content'])
    return content

def gemini_response(prompt):
    #google API Key
    google_api_key = config["LLM model"]["gemini_api_key"]
    genai.configure(api_key=google_api_key, transport = 'rest')
    model = genai.GenerativeModel('models/gemini-1.5-flash')

    response = model.generate_content([prompt]).text

    return response

def llm_response(prompt):
    if config["LLM model"]["model"] == "gpt-4o":
        return gpt4_response(prompt)
    elif config["LLM model"]["model"] == "gemini-1.5":
        return gemini_response(prompt)
    else:
        raise ValueError("Unsupported model specified in config.json")