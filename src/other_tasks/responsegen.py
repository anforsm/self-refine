from openai_wrapper import call_openai
import pandas as pd
from tqdm import tqdm
import json

def init(context):
  question_prefix = "Conversation history: "
  answer_prefix = "Response: "
  intra_example_sep="\n\n"
  inter_example_sep="\n\n### END ###\n\n"

  instruction = "Provided a dialogue between two speakers, generate a response that is coherent with the dialogue history. Desired traits for responses are: 1) Relevant - The response addresses the context, 2) Informative - The response provides some information, 3) Interesting - The response is not interesting, 4) Consistent - The response is consistent with the rest of the conversation in terms of tone and topic, 5) Helpful - The response is helpful in providing any information or suggesting any actions, 6) Engaging - The response is not very engaging and does not encourage further conversation, 7) Specific - The response contains pecific content, 9) User understanding - The response demonstrates an understanding of the user's input and state of mind, and 10) Fluent. Response should begin with - Response:\n\n"

  # loop through a list of conversation histories?
  query = f"""Conversation history

{history}

Response: {response}"""

  prompt = f"""{instruction}{inter_example_sep.join(query)}{inter_example_sep}"""

  final_query = f"{prompt}{question_prefix}\n\n{context}{intra_example_sep}"
  return call_openai(final_query), final_query

def iterate(response_to_scores):
  pass