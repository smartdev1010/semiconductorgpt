from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

df = pd.read_excel("./xlsxs/amd_financial_statement.xlsx")
df = df.transpose()
print(df.describe())
amd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/intc_financial_statement.xlsx")
df = df.transpose()
print(df.describe())
intc_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/nvda_financial_statement.xlsx")
df = df.transpose()
print(df.describe())
nvda_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/tsm_financial_statement.xlsx")
df = df.transpose()
print(df.describe())
tsm_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

tools = [
    Tool(
        name="AMD Company Data",
        func=amd_agent.run,
        description="useful for when you need to answer questions about AMD company financial statements"
    ),
    Tool(
        name="Intel Company Data",
        func=intc_agent.run,
        description="useful for when you need to answer questions about Intel company financial statements"
    ),
    Tool(
        name="Nvidia Company Data",
        func=nvda_agent.run,
        description="useful for when you need to answer questions about Nvidia company financial statements"
    ),
    Tool(
        name="TSMC Company Data",
        func=tsm_agent.run,
        description="useful for when you need to answer questions about TSMC company financial statements"
    )
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run("Compare Intel with its competitors financially.")