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
    prompt = f'''You are an AI, friend of deepak, he  designed you to help people by providing detailed and accurate answers. 
                Converse like a human and short and engaging manner. 
Note: 
your role is to help people know about deepak!!!
                Here is some information about me to help you answer my question:
                

DEEPAKSAKTHI VELLORE KUMAR
ML, Data Science, Language Model Enthusiast +91 9787558677

q deepaksakthi-v-k-4a7fe.web.app  Chennai, India
EXPERIENCE Conversational Chatbot Developer Intern Bimbasree Private Limited 06/2024 - Present
Chennai, India xtendlabs.com
Responsible for contributing to the development and implementation of AI models, participating in research and development activities, and assisting in the creation of innovative Gen AI solutions.
AI/ML Engineer Intern ToToys AI 06/2024 - Present
 deepak2004sakthi@gmail.com q linkedin.com/in/deepaksakthi-v-k
PROJECTS RAG - Personal Chatbot lasttry-7irc.onrender.com
Launched a Retrieval Augmented Generative Model to answer user questions using external data. Utilized Langchain, crewAI and vector database for implementation.
Character Level GPT Model Banglore, India - Remote
Developing an AI-powered products for clients, aiming to increase user engagement. Fullstack AI Developer Intern KoworkerAI 04/2024 - Present
Dubai, UAE - Remote
• •
Minimised recruitment time by 50% through an internal tool built with Python, Flask, and CrewAI LinkedIn API. Improved candidate selection accuracy by 30%, and shortened time-to-hire by 25% through the implementation of language models for job description generation and candidate evaluation.
Freelance Data Analyst 2023
Under the guidance of Dr. Lokesh (Mch Neurosurgery), conducted detailed retrospective analysis on a patient cohort to assess functional and surgical outcomes. This involved rigorous data analysis and interpretation to identify critical patterns and trends. These insights were pivotal in prognosticating future patient outcomes and optimizing treatment strategies.
Data Science Trainee Innomatics Research Labs 02/2023 - 05/2023
Implemented a Character-Level Generative Pre-trained Transformer (GPT) model from scratch, achieving a model size of approximately 0.212 million parameters.
Git Audio/Video Summarizer
Developed a comprehensive audio extraction and transcription solution for videos/audios using Whisper and Gemini Pro; streamlined key point comprehension and information retrieval, boosting productivity by 35% across the team.
Git MLP - implementation
Implemented character-level Multilayer Perceptrons (MLPs) inspired by the research paper by Bengio. The aim is to predict the next character or word in a given sequence.
Git ANN’s Back propagation Implementation
A streamlined neural network implementation, creating a lightweight Autograd engine and neural networks library; improved computational performance by 50% and reduced code base complexity by 20%.
Git Hyderabad, India - Remote • •
Built efficient APIs using Flask and Streamlit, reducing response time by 25%, optimised database queries with SQLAlchemy and SQL, enhancing data retrieval speed by 30%. Deployed machine learning models on cloud, achieving 92% accuracy in classification,regression tasks.
Student Intern Edunet Foundations 11/2022
Personal Portfolio Website deepaksakthi-v-k-4a7fe.web.app
Designed a website with HTML, CSS, JavaScript, and Bootstrap.
Full stack Movie Review Application
Developed a full-stack web application using Java Spring Boot, React, and MongoDB.
Git Bangalore, India - Remote
Launched an AI-powered verification system for diabetes patients, analyzing diagnostic parameters to improve early diagnosis accuracy by 25% and patient outcomes.
Data Science Student Intern DevTown 04/2022 - 08/2022
Bangalore, India -Remote
Completed 15-weeks of intensive training on machine learning, data science, and deep learning.
EDUCATION Bachelor in Information Technology
St. Joseph College of Engineering 2021 - Present
Chennai, India
High School Sri Chaitanya Techno Schools 2019 - 2021
SKILLS P rogramming Languages Python | Java | C | Processing(Basics) AI Integration Frameworks Langchain | crewAI | LlamaIndex Backend tools and Frameworks Flask | Django | SpringBoot(Basics) | NodeJS(Basics) Database
8.19 10.0
GPA /
Percentage 93.8 100
Bangalore, India /
CERTIFICATION B2 Business Vantage | Cambridge English Certified in Advanced Programming in Bharathiar University - CCII programme
Machine Learning with Python - IBM, Coursera Statistics for Data Science, Coursera
MariaDB | PostgreSQL | MongoDB Setting and configuring Database Clusters(Galera) and Replications(Master-Slave | Master-Master)
Operating Systems Windows NT/ 2000/ XP/ Vista, Linux, Unix Additional Tools
Machine Learning, Deep Learning,Scikit-Learn, PyTorch, TensorFlow, Flask, NumPy, Pandas,Matplotlib, Seaborn, Keras, NLTK (Natural Language Toolkit), Javascript, HTML/CSS, OpenCV AWS(EC2, VPC,S3,ELB), docker(Basics) Configuring OpenLDAP, Ngnix for load balancing

Answer as you are my friend!!!
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
