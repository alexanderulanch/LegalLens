import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import openai
import gradio as gr

model_engine = "text-embedding-ada-002"

location_info = {
    "Boulder": {
        "npz_file": "embeddings/boulder_embeddings.npz",
        "role_description": "You are an AI-powered legal assistant specializing in the jurisdiction of Boulder County, Colorado."
    },
    "Denver": {
        "npz_file": "embeddings/denver_embeddings.npz",
        "role_description": "You are an AI-powered legal assistant specializing in the jurisdiction of Denver, Colorado."
    }
}


def generate_matches(query, location, api_key):
    """
    Generate legal information matches based on the user's query and location.

    This function uses OpenAI's GPT-4 and text-embedding models to generate
    legal information matches for a given query and location. The function
    returns an HTML response containing the AI-generated response and references
    to relevant legal information.

    Parameters
    ----------
    query : str
        The user's query for legal information.
    location : str
        The location for which the user is seeking legal information.
        Possible values are "Boulder" and "Denver".
    api_key : str
        The OpenAI API key required to access the GPT-4 and text-embedding models.

    Returns
    -------
    str
        An HTML response containing the AI-generated response and references
        to relevant legal information. In case of an error, an error message
        is returned in HTML format.
    """
    try:
        openai.api_key = api_key
        query_embedding = openai.Embedding.create(
            input=query,
            model=model_engine
        )

        query_embedding_json = query_embedding.to_dict()
        query_embedding = np.array(
            query_embedding_json['data'][0]['embedding'])

        # Determine the .npz file and role description to use based on the user-selected location
        location_data = location_info.get(location)
        if not location_data:
            raise Exception(f"No data found for location '{location}'.")

        npz_file = location_data['npz_file']
        role_description = location_data['role_description']

        data = np.load(npz_file, allow_pickle=True)
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

        role = f'''
          {role_description} 
          Your expertise lies in providing accurate and timely information on the laws and regulations specific to your jurisdiction.

          Your role is to assist individuals, including law enforcement officers, legal professionals, and the general public, 
          in understanding and applying legal standards within this jurisdiction. You are knowledgeable, precise, and always 
          ready to offer guidance on legal matters. Your max_tokens is set to 120 so keep your
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

    except Exception as e:
        html_message = '''<p style="font-family: Arial, sans-serif; font-size: 16px; color: #333;">
    <strong>Notice:</strong> An error occurred while processing your request. Please see the details below:
</p>'''
        return f"<p>{html_message} {str(e)}</p>"


description = "LegalLens is an AI-powered legal research app designed to assist individuals, including law enforcement officers, legal professionals, and the general public, in accessing accurate legal information. The app covers various jurisdictions and ensures that users can stay informed and confident, regardless of their location. This demo is meant to serve as a proof of concept."

iface = gr.Interface(title="LegalLens Demo", description=description, fn=generate_matches, inputs=[gr.Textbox(label="Query"), gr.Dropdown(choices=["Boulder", "Denver"], label="Location"), gr.Textbox(label="OpenAI API key", placeholder="must have access to GPT-4")], outputs=["html"], examples=[
    ["Is it legal for me to use rocks to construct a cairn in an outdoor area?", "Boulder"],
    ["Is it legal to possess a dog and take ownership of it as a pet in Boulder?", "Boulder"],
    ["As per the local laws, am I allowed to expose my upper body and go without a shirt in public places?", "Boulder"],
    ["What are the legal restrictions regarding the maximum height of a building that can be constructed in a particular area?", "Boulder"],
    ["Is it legal to place a couch on my porch according to the local regulations?", "Boulder"],
    ["Can I legally graze my llamma on public land?", "Boulder"]
])

iface.launch()
