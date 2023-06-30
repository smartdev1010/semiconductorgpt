from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import asyncio
import pinecone
import openai
import os
import json
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

ans_chatgpt = ""
ans_docqa = ""
ans_googlegpt = ""

openai.api_key = os.getenv("OPENAI_API_KEY")


async def chatgpt(query: str):
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


index_name = "semiconduct-retrieval"
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)


async def docqa(query: str):
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


async def googlegpt(query):
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


async def main(query: str):
    tasks = [
        asyncio.ensure_future(chatgpt(query)),
        asyncio.ensure_future(docqa(query)),
        asyncio.ensure_future(googlegpt(query)),
    ]
    await asyncio.gather(*tasks)
    print("\nAll functions completed\n")
    print(
        "\n=====================================================================================\n"
    )


if __name__ == "__main__":
    while True:
        query = input("Input your query: ")

        # functions = [
        #     {
        #         "name": "check_and_run_visualization",
        #         "description": "Check message requires graph and run proper actions.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "graph_required": {
        #                     "type": "boolean",
        #                     "description": "Check message requires graph and run proper actions. If required, true else, false.",
        #                 }
        #             },
        #             "required": ["graph_required"],
        #         },
        #     }
        # ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI Bot that determines given query requires graph visualization answer. For example, if query asks you to compare companies financial data, it requires graph visualization. Only return 'yes' or 'no' depends on it requires graph.",
                },
                {"role": "user", "content": f"This is query: {query}."},
            ],
            # functions=functions,
            # function_call="auto",
        )

        message = response["choices"][0]["message"]
        print(message)

        if message.content == "No":
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main(query))

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are You are a McKinsey partner who is known for his cutting edge insights. The stakes are high & you are consulting a client who is going to give you a 100 million contract if you are insightful enough. You always give a so-what to the client when providing facts. You never give random answers that have no meaning and you are always focused on nuanced insights combining multiple legitimate sources of information. DON'T mention you are combining several source of information in the answer.",
                    },
                    {
                        "role": "user",
                        "content": f"Question is {query}.\nYou have two answers to pick from as sources of insights .\nFirst optional answer is {ans_chatgpt}. \nSecond optional answer is {ans_docqa}.\nThird optional answer is {ans_googlegpt}\n\nPlease combine & synthesize all the answers in a way that the context from all the answers is maintained and return the combined insights as the final answer.",
                    },
                ],
            )
            ans_final = completion.choices[0].message.content

            print(ans_final)

            print(
                "\n=====================================================================================\n"
            )
        else:
            df = pd.read_excel("./xlsxs/Merged data.xlsx")
            df = df.astype(str)
            xlsx_string = df.to_csv(index=False, sep="\t")

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a McKinsey consultant analyzing financial information.",
                    },
                    {
                        "role": "user",
                        "content": f"Following is semiconductor companies financial tabular information.\n\n {xlsx_string} \n\n {query}",
                    },
                ],
            )

            anaylze = completion["choices"][0]["message"].content

            print(anaylze)

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a McKinsey consultant analyzing financial information.",
                    },
                    {
                        "role": "user",
                        "content": f"Following is analyzed financial information.\n\n {anaylze} \n\n Give me python code that visualize above information by using matplotlib library. Only return code.",
                    },
                ],
            )

            code = completion["choices"][0]["message"].content

            code = code.replace("python", "")
            if "```" in code:
                extract = code.split("```")[1]
            else:
                extract = code

            print(extract)

            exec(extract)
