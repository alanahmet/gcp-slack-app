import os
import asyncio
import asyncpg
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pgvector.asyncpg import register_vector
from google.cloud.sql.connector import Connector
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

load_dotenv(find_dotenv())


def draft_email(user_input, recipient_name="Dave") -> str:
    """
    Generate a draft email reply based on user input.

    Parameters:
        user_input (str): The content of the user's email to reply to.
        recipient_name (str, optional): The name of the person to address in the email.
                                        Defaults to "Dave".

    Returns:
        str: The drafted email reply.
    """
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    You are a helpful assistant that drafts an email reply based on a new email.
    Your goal is to help the user quickly create a perfect email reply.
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    Make sure to sign off with {signature}.
    """

    signature = f"Kind regards, \n\{recipient_name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=recipient_name)

    return response


def get_embedded_query(creative_prompt: str) -> list:
    """
    Get vector embeddings for a creative prompt.

    Parameters:
        creative_prompt (str): The creative prompt for which vector embeddings are required.

    Returns:
        list: A list of vector embeddings for the given creative prompt.
    """
    embeddings_service = VertexAIEmbeddings()
    qe = embeddings_service.embed_query([creative_prompt])
    return qe


async def get_vector_search_from_cloudsql(qe: list) -> list:
    """
    Perform vector search using Cloud SQL and retrieve matches based on vector embeddings.

    Parameters:
        qe (list): List of vector embeddings to be used for similarity search.

    Returns:
        list: List of descriptions of products matching the vector search.
    """
    matches = []
    project_id = os.environ["project_id"]
    region = os.environ["region"]
    instance_name = os.environ["instance_name"]
    database_user = os.environ["database_user"]
    database_password = os.environ["database_password"]
    database_name = os.environ["database_name"]
    
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await register_vector(conn)
        similarity_threshold = 0.7

        # Find similar products to the query using cosine similarity search
        # over all vector embeddings. This new feature is provided by `pgvector`.
        results = await conn.fetch(
            """
            WITH vector_matches AS (
                SELECT product_id, 1 - (embedding <=> $1) AS similarity
                FROM product_embeddings
                WHERE 1 - (embedding <=> $2) > $3
                ORDER BY similarity DESC
                LIMIT 1
            )
            SELECT description FROM products
            WHERE product_id IN (SELECT product_id FROM vector_matches)
            """,
            qe,
            qe,
            similarity_threshold,
        )

        for r in results:
            matches.append(r["description"])

        await conn.close()
        return matches


def generate_creative_output(matches: list, creative_prompt: str) -> str:
    """
    Generate creative output based on product descriptions and a creative prompt.

    Parameters:
        matches (list): List of product descriptions for generating creative output.
        creative_prompt (str): The creative prompt to be used for generating the output.

    Returns:
        str: The generated creative output based on the provided inputs.
    """
    template = """
        You are given descriptions about some similar kind of toys in the context.
        This context is enclosed in triple backticks (```).
        Combine these descriptions and adapt them to match the specifications in
        the initial prompt. All the information from the initial prompt must
        be included. You are allowed to be as creative as possible,
        and describe the new toy in as much detail. Your answer should be
        less than 200 words.

        Context:
        ```{context}```

        Initial Prompt:
        {creative_prompt}

        Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "creative_prompt"])

    openai_api_key = os.environ["openai_api_key"]
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    answer = llm_chain.run(
        {
            "context": "\n".join(matches),
            "creative_prompt": creative_prompt,
        }
    )

    return answer
