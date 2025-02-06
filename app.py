from flask import Flask, request, send_file
import io
from flask_cors import CORS  # Necessary for cross-domain requests

app = Flask(__name__)
CORS(app)  # This is used to allow cross-domain requests during development

from groq import Groq
from transformers import pipeline

# Groq API Key
client = Groq(api_key="Your_Groq_Key")

# Hugging Face pipeline for TTS (Text-to-Speech)
tts_pipeline = pipeline("text-to-speech", model="facebook/tts_transformer-es-css10")

# Initialize conversation history with instructions for the AI model
INIT_MESG = [
    {
        "role": "user",
        "content": """
            You are a voice chatbot that responds to human user's speech input.
            The speech input texts are sometimes broken or hard stop due to the listening mechanism,
            If the message you read is not complete, please ask the user to repeat or complete politely and concisely.
            Remember you are speaking not writing, so please use oral expression in plain language.
        """
    },
    {
        "role": "assistant",
        "content": "OK, I understood.",
    },
]

history_messages = INIT_MESG

@app.route('/synthesize-speech', methods=['POST'])
def synthesize_speech():
    data = request.json
    text = data['text']

    # Use Hugging Face's TTS pipeline to generate speech from text
    audio = tts_pipeline(text)

    # Save the audio to a buffer and send it back to the client
    audio_path = "output.wav"
    with open(audio_path, "wb") as f:
        f.write(audio['speech'])

    return send_file(
        audio_path,
        mimetype="audio/wav",
    )

@app.route('/process-speech', methods=['POST'])
def process_speech():
    data = request.json
    user_text = data['text']
    history_messages.append({"role": "user", "content": user_text})

    # Call the Groq API for a response
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=history_messages,
    )

    ai_response = completion.choices[0].message.content
    history_messages.append({"role": "assistant", "content": ai_response})

    return {'response': ai_response}

@app.route('/start-speech', methods=['POST'])
def start_speech():
    global history_messages
    history_messages = INIT_MESG
    return {'response': 'OK'}

if __name__ == '__main__':
    app.run(debug=True)
