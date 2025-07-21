from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
app = Flask(__name__)

# Initialize LLM + Memory
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    system_instruction = (
        "You are a helpful and responsible health assistant. "
        "You can provide general health information and advice on lifestyle, diet, exercise, "
        "symptom explanations, and when to see a doctor. "
        "Always remind the user that your responses are not medical advice and they should "
        "consult a qualified healthcare professional for serious concerns."
    )
    full_input = f"{system_instruction}\n\nUser: {user_message}"
    response = conversation.predict(input=full_input)
    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(debug=True)

