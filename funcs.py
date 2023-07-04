import openai
from dotenv import load_dotenv
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEy")


def chatgpt(query: str):
    print("\nChatGPT answer\n")
    print(
        "\n=====================================================================================\n"
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a McKinsey partner who is known for his cutting edge insights. You are consulting a client who is going to give you a 100 million contract if you are insightful enough. You always give a so-what to the client when providing facts. You never give random answers that have no meaning and you are always focused on nuanced insights combining multiple legitimate sources of information. .",
            },
            {"role": "user", "content": query},
        ],
    )
    ans_docqa = completion.choices[0].message.content
    print(ans_docqa)
    print("\nChatGPT answer completed\n")
    print(
        "\n=====================================================================================\n"
    )
    return ans_docqa


index_name = "semiconduct-retrieval"
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)


def docqa(query: str):
    print("\nAnswer based on Docs\n")
    print(
        "\n=====================================================================================\n"
    )
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name, embedding=OpenAIEmbeddings()
    )
    # vectorstore = FAISS.load_local("./vector", embeddings=OpenAIEmbeddings())
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    result = qa({"query": query})
    ans_docqa = result["result"]
    print(ans_docqa)
    print(f"\nSOURCE ---------------->\n")
    print(
        "\n=====================================================================================\n"
    )
    for source_doc in result["source_documents"]:
        print(source_doc.metadata["source"])
    print(
        "\n===================================================================================\n"
    )
    print("\nDocQA completed\n")
    print(
        "\n=====================================================================================\n"
    )

    return ans_docqa


def googlegpt(query):
    print("\nGoogleGPT started\n")
    print(
        "\n=====================================================================================\n"
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    search = GoogleSearchAPIWrapper()
    tools = [
        Tool.from_function(
            func=search.run,
            name="Search",
            description="useful for when you need to answer questions about current events",
        )
    ]
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    ans_googlegpt = agent_chain.run(query)
    print(ans_googlegpt)
    print("\nGoogleGPT completed\n")
    print(
        "\n=====================================================================================\n"
    )

    return ans_googlegpt
