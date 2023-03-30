import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import openai
import gradio as gr
import json

model_engine = "text-embedding-ada-002"


def generate_matches(query, api_key):
    openai.api_key = api_key
    query_embedding = openai.Embedding.create(
        input=query,
        model=model_engine
    )
    query_embedding_json = query_embedding.to_dict()
    query_embedding = np.array(query_embedding_json['data'][0]['embedding'])
    data = np.load('/content/jurisdiction_data_embeddings.npz',
                   allow_pickle=True)
    embeddings = data['embeddings']

    jurisdiction_data = pd.DataFrame({
        'url': data['urls'],
        'title': data['titles'],
        'subtitle': data['subtitles'],
        'content': data['contents']
    })

    distances = cdist(query_embedding.reshape(1, -1),
                      embeddings, metric='cosine')[0]
    indices = np.argsort(distances)[:3]
    top_matches = jurisdiction_data.iloc[indices].to_dict('records')

    role = '''
      You are an AI-powered legal assistant specializing in the jurisdiction of
      Boulder County, Colorado. Your expertise lies in providing accurate and timely
      information on the laws and regulations specific to Boulder.

      Your role is to assist law enforcement officers in understanding and applying
      legal standards within this jurisdiction. You are knowledgeable, precise, and
      always ready to offer guidance on legal matters pertaining to Boulder, Colorado.
      '''

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": role.strip()},
            {"role": "system", "content": str(top_matches)},
            {"role": "user", "content": query},
            {"role": "assistant", "content": ""},
        ],
        temperature=0.7,
        max_tokens=120,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message = response["choices"][0]["message"]["content"].strip()
    html_response = "<p><strong>Response:</strong></p><p>" + message + "</p>"
    html_references = "<p><strong>References:</strong></p><ul>"
    for match in top_matches:
        html_references += f'<li><a href="{match["url"]}">{match["title"]}: {match["subtitle"]}</a></li>'
    html_references += "</ul>"

    return html_response + html_references


description = "LawLens Boulder County is an AI-powered legal research app specifically for law enforcement officers in Boulder County, Colorado. With quick access to accurate information, officers can stay informed and confident while on the job. This demo is meant to serve as a proof of concept."

iface = gr.Interface(title="LawLens Boulder County Demo", description=description, fn=generate_matches, inputs=[gr.Textbox(label="Query"), gr.Textbox(label="OpenAI API key", placeholder="must have access to GPT-4")], outputs=["html"], examples=[["Can I conduct a search of a vehicle if I smell marijuana coming from the car?"],
                                                                                                                                                                                                                                                     ["What are the requirements for conducting a traffic stop in this jurisdiction?"],
                                                                                                                                                                                                                                                     ["Under what circumstances can I perform a warrantless arrest in this jurisdiction?"],
                                                                                                                                                                                                                                                     ["What are the guidelines for using force in self-defense as a law enforcement officer in this jurisdiction?"],
                                                                                                                                                                                                                                                     ["Can a person openly carry a firearm in public spaces in this jurisdiction?"],
                                                                                                                                                                                                                                                     ["When is it permissible to use a taser during an arrest in this jurisdiction?"],
                                                                                                                                                                                                                                                     ["What constitutes probable cause for a search in this jurisdiction?"],
                                                                                                                                                                                                                                                     ["What are the protocols for handling domestic violence situations in this jurisdiction?"],
                                                                                                                                                                                                                                                     ])


iface.launch(debug=True)
