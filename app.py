from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import warnings
from dotenv import load_dotenv
import os
import requests
import ssl 
import textwrap
from IPython.display import display
from IPython.display import Markdown
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


genai.configure(api_key="AIzaSyD7tk9HKuwLK5t3_NOvvlUCLREYRgIIDPE")

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Set the SSL context to avoid verification issues within the Flask app context
ssl._create_default_https_context = ssl._create_unverified_context


def prettify_text(text):
    prettified = text.replace('**', '\n').replace('*', '\n')
    return prettified    

def get_google_api_key():
    return os.getenv("AIzaSyD7tk9HKuwLK5t3_NOvvlUCLREYRgIIDPE")

# Import gevent and monkey-patch early to avoid MonkeyPatchWarning
import gevent.monkey
gevent.monkey.patch_all()

def llm_model(question, data):
    model = genai.GenerativeModel('gemini-1.5-pro')
    logger.info("-------------------------DATA PASSING TO THE MODEL!!!--------------------------")
    prompt = f'''You are an AI chatbot designed to help people by providing detailed and accurate answers based on context about your friend deepak. You are more than an assistant; you are a friend to Deepak. Ensure your responses are informative, contextually relevant, and align with people's tone and style of communication.
                Converse like a human
                Context:\n{data}\n
                
                Question:\n{question}\n
                
                Note:
                - If the question is directly related to the provided data, provide a detailed and accurate answer.
                - If the question pertains to a general topic or is conversational in nature, respond in a friendly, human-like manner. For example, for questions like "tell me something you know" or "is this the right time to talk," answer conversationally and encourage engagement.
                - If the answer is not present in the provided context and the question seems personal, unknown, or too common, respond by acknowledging your limitation in a friendly way. For instance, say "Oh no! I am not really aware of it, I shall ask Deepak and let you know later!!" to avoid giving incorrect information.
                - Always prioritize clarity, accuracy, and a friendly tone in your responses. Even if the answer is not relevant, try to respond in a helpful and engaging manner.
                - Always answer in short and precisely!!!
                
                Answer:'''

    response = model.generate_content(prompt)    
    logger.info("-------------------------MODEL DATA DONE!!!--------------------------\n\n\n\n\n")            
    return response.text

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def user_input(user_question):
    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyBAfHsxOVghV_vFUwxwwQwV68Agpq4av80')
    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("-------------------------DATABASE LOADED!!!--------------------------")    
    # Search for similar documents
    docs = new_db.similarity_search(user_question,k=3)
    logger.info("-------------------------RETRIEVED SIMILAR DATA!!!--------------------------") 
    context = " ".join([doc.page_content for doc in docs])
    print(context)

    return context

@app.route('/')
def home():
    return "Welcome to my ai's backend, well done u hacker!!"

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_question = data.get('message', '')
        logger.info(f"USER QUESTION: {user_question}")
        
        response = user_input(user_question)
        out = llm_model(user_question, response)
        logger.info(f"User Question: {user_question}, Response: {out}")

        return jsonify({'response': prettify_text(out)})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': 'An error occurred, please try again later.'}), 500


# #Define the ask route to handle POST requests
# @app.route('/ask', methods=['POST'])
# def ask():
#     # Get user's question from the request
#     user_question = request.form['question']
#     logger.info(f"USER QUESTION: {user_question}")
    
#     # Get response based on user's question
#     response = user_input(user_question)
#     out = llm_model(user_question, response)
#     logger.info(f"User Question: {user_question}, Response: {out}")

#     # Return the response as JSON
#     return jsonify({'response': prettify_text(out)})



# Run the Flask app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0')


