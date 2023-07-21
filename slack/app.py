import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from functions import get_embeded_query, get_vector_search_from_cloudesql, generate_createive_output
import asyncio

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

# Initialize llm memory component
memory=ConversationBufferWindowMemory(k=2)


def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    emma_info = "EMMA is an AI power gift recommender chatbot. You can search items from Walmart using /search-walmart commend "
    say(emma_info)
    
    
@app.event("message")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]
    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()  
    
    template = """You are a gift recommender chatbot. Your mission is to give people gift ideas based on their preferences 
    {history}
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    conversation = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=os.environ["openai_api_key"]),
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    response = conversation.predict(human_input=text)
    say(response)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# The echo command simply echoes on command
@app.command("/search-walmart")
def repeat_text(ack, respond, command):
    """
    Route for handling Slack command.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    respond("Searching database please wait")
    ack()    
    text = f"{command['text']}"
    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()    
    query = get_embeded_query(text)
    search_res =  asyncio.run(get_vector_search_from_cloudesql(query)) 
    response = generate_createive_output(search_res, text)
    respond(response)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run(debug=True)

# %%
