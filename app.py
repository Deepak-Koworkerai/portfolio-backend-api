from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import warnings
from dotenv import load_dotenv
import os
import ssl 
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the SSL context to avoid verification issues within the Flask app context
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to prettify text
def prettify_text(text):
    prettified = text.replace('**', '\n').replace('*', '\n')
    return prettified    

# Define the ask route to handle POST requests
@app.route('/')
def home():
    return "Welcome to my ai's backend, well done u hacker!!"

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_question = data.get('message', '')
        logger.info(f"USER QUESTION: {user_question}")

        # Use OpenAI model to get response
        out = llm_model(user_question)
        logger.info(f"User Question: {user_question}, Response: {out}")

        return jsonify({'response': prettify_text(out)})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': 'An error occurred, please try again later.'}), 500

def llm_model(question):
    logger.info("-------------------------DATA PASSING TO THE MODEL!!!--------------------------")
    
    AZURE_OPENAI_API_KEY = 'ef29eaa3ecd04bc1b582e40759708c9e'
    AZURE_OPENAI_ENDPOINT = 'https://langchain-poc.openai.azure.com/'
    AZURE_MODEL = 'langchain-poc-gpt35-turbo-0125'

    # Create the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-02-01"
    )

    # Construct the prompt
    prompt = f'''You are an AI chatbot designed to help people by providing detailed and accurate answers. 
                Converse like a human. 

                Here is some information about me to help you answer my question:
                Name: Deepak Sakthi V
                Location: Chennai, India
                Phone: +91 9787558677
                Email: deepak2004sakthi@gmail.com
                LinkedIn: linkedin.com/in/deepaksakthi-v-k

                Experience:
                - Conversational Chatbot Developer Intern at Bimbasree Private Limited (06/2024 - Present)
                  - Contributing to AI models and innovative Gen AI solutions.
                - AI/ML Engineer Intern at ToToys AI (06/2024 - Present)
                  - Developing AI-powered products to enhance user engagement.
                - Fullstack AI Developer Intern at KoworkerAI (04/2024 - Present)
                  - Minimizing recruitment time and improving candidate selection with AI models.

                Projects:
                - RAG - Personal Chatbot: Launched a Retrieval Augmented Generative Model.
                - Character Level GPT Model: Developing AI products for clients.
                - AI-powered verification system for diabetes patients.

                Skills:
                - Programming Languages: Python, Java, C
                - AI Integration Frameworks: Langchain, crewAI, LlamaIndex
                - Database Management: MariaDB, PostgreSQL, MongoDB
                - Machine Learning Frameworks: Scikit-Learn, TensorFlow, PyTorch
                - Web Development: Flask, Django, SpringBoot, JavaScript, HTML/CSS

                Question:\n{question}\n
                
                Answer:'''

    # Call Azure OpenAI API for response
    try:
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful friend."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response from Azure OpenAI: {e}")
        answer = "Sorry, I encountered an error while processing your request."

    logger.info("-------------------------MODEL DATA DONE!!!--------------------------\n\n\n\n\n")            
    return answer

# Run the Flask app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0')
