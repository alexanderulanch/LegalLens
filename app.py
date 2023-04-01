import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import openai
import gradio as gr
import json

model_engine = "text-embedding-ada-002"


def generate_matches(query, location, api_key="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"):
    try:
        openai.api_key = api_key
        query_embedding = openai.Embedding.create(
            input=query,
            model=model_engine
        )

        query_embedding_json = query_embedding.to_dict()
        query_embedding = np.array(
            query_embedding_json['data'][0]['embedding'])
        data = np.load('jurisdiction_data_embeddings.npz', allow_pickle=True)
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
          You are an AI-powered legal assistant specializing in the jurisdiction of Boulder County, Colorado. 
          Your expertise lies in providing accurate and timely information on the laws and regulations specific to Boulder.

          Your role is to assist individuals, including law enforcement officers, legal professionals, and the general public, 
          in understanding and applying legal standards within this jurisdiction. You are knowledgeable, precise, and always 
          ready to offer guidance on legal matters pertaining to Boulder, Colorado. Your max_tokens is set to 120 so keep your
          response below that.
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

    except:
        html_message = '''<p style="font-family: Arial, sans-serif; font-size: 16px; color: #333;">
    <strong>Notice:</strong> The OpenAI API key is either invalid or does not have access to GPT-4. To create a valid API key, please visit the following link: 
    <a href="https://platform.openai.com/account/api-keys" target="_blank">https://platform.openai.com/account/api-keys</a>
</p>'''
        return f"<p>{html_message} {api_key}</p>"


description = "LegalLens is an AI-powered legal research app designed to assist individuals, including law enforcement officers, legal professionals, and the general public, in accessing accurate legal information. The app covers various jurisdictions and ensures that users can stay informed and confident, regardless of their location. This demo is meant to serve as a proof of concept."

iface = gr.Interface(title="LegalLens Demo", description=description, fn=generate_matches, inputs=[gr.Textbox(label="Query"), gr.Dropdown(choices=["Boulder County, Colorado", "San Francisco, California"], label="Location"), gr.Textbox(label="OpenAI API key", placeholder="must have access to GPT-4")], outputs=["html"], examples=[
    ["What resources are available to support individuals in a mental health crisis?"],
    ["What are the regulations for noise levels in residential areas?"],
    ["What are the parking restrictions in downtown areas?"],
    ["How can I appeal a parking ticket?"],
    ["What are the requirements for obtaining a business license?"],
    ["What are the guidelines for recycling and waste disposal?"],
    ["How can I report a pothole or damaged road?"],
    ["What are the regulations for leash laws and pet ownership?"]
])

iface.launch(debug=True)
