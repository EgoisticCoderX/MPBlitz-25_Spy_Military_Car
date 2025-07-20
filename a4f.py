import requests
import json
from openai import OpenAI

A4F_API_KEY = "ddc-a4f-54380a13389f4590a8e791abd61c48ec"
A4F_BASE_URL = "https://api.a4f.co/v1"

MODEL_ID = "provider-5/gpt-4o"

client = OpenAI(
  api_key=A4F_API_KEY,
  base_url=A4F_BASE_URL
)

response = client.chat.completions.create(
  model=MODEL_ID,
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
          }
        }
      ]
    }
  ]
)

print(response.choices[0].message.content)