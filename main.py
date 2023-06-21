from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# loader = PyPDFDirectoryLoader("./docs")
# documents = loader.load_and_split(CharacterTextSplitter(chunk_size=1000, chunk_overlap=256))

# vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
# vectorstore.save_local("./vector")
vectorstore = FAISS.load_local("./vector", embeddings=OpenAIEmbeddings())

query = "Give me a perspective on the Semiconductor industry. What is the future outlook of this industry? What are the drivers of this market? Tell me what capabilities does a player need to win in this market? Keep in mind the global interconnecting nature of the players and countries in this industry."

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0), vectorstore.as_retriever(), memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))
result = qa({"question": query})
answer1 = result["answer"]

print("\nAnswer based on Docs\n")
print("\n=====================================================================================\n")
print(answer1)
print("\n=====================================================================================\n")

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
    {"role": "system", "content": "You are a McKinsey partner who is known for his cutting edge insights. You are consulting a client who is going to give you a 100 million contract if you are insightful enough. You always give a so-what to the client when providing facts. You never give random answers that have no meaning and you are always focused on nuanced insights combining multiple legitimate sources of information. ."},
    {"role": "user", "content": query}
  ],
  
)
answer2 = completion.choices[0].message.content

print("\nChatGPT answer\n")
print("\n=====================================================================================\n")
print(answer2)
print("\n=====================================================================================\n")

prompt_template = """You are You are a McKinsey partner who is known for his cutting edge insights. The stakes are high & you are consulting a client who is going to give you a 100 million contract if you are insightful enough. You always give a so-what to the client when providing facts. You never give random answers that have no meaning and you are always focused on nuanced insights combining multiple legitimate sources of information. 

Question is {query}
You have two answers to pick from as sources of insights .
First optional answer is {answer1}.
Second optional answer is {answer2}.

If first optional answer is kind of 'I don't know' then just return second optional answer as final.
If not, please combine & synthesize both the answers in a way that the context from both the answers is maintained and return the combined insights as the final answer."""

llm_chain = LLMChain(llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, max_tokens=7500), prompt=PromptTemplate.from_template(prompt_template))
final = llm_chain.run(query=query, answer1=answer1, answer2=answer2)

print("\Final answer\n")
print("\n=====================================================================================\n")
print(final)
print("\n=====================================================================================\n")
