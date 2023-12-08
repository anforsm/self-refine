from openai import OpenAI
from openai_key import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def call_openai(prompt):
  print("sent request to openai")
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    #model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.01,
    max_tokens=2000,
  )
  print(response)
  print("got response from openai")
  return response.choices[0].message.content
  