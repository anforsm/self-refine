from openai import OpenAI
from openai_key import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def call_openai(prompt):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=300,
  )

  return response.choices[0].message.content
  