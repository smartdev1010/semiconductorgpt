from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

df = pd.read_excel("./xlsxs/amat_financial_statement.xlsx")
amat_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/amd_financial_statement.xlsx")
amd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/asml_financial_statement.xlsx")
asml_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/avgo_financial_statement.xlsx")
avgo_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/intc_financial_statement.xlsx")
intc_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/lrcx_financial_statement.xlsx")
lrcx_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/mu_financial_statement.xlsx")
mu_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/nvda_financial_statement.xlsx")
nvda_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/qcom_financial_statement.xlsx")
qcom_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/ssnlf_financial_statement.xlsx")
ssnlf_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/tsm_financial_statement.xlsx")
tsm_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

df = pd.read_excel("./xlsxs/txn_financial_statement.xlsx")
txn_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

tools = [
    Tool(
        name="AMAT Company Data",
        func=amat_agent.run,
        description="useful for when you need to answer questions about AMAT company financial statements"
    ),
    Tool(
        name="AMD Company Data",
        func=amd_agent.run,
        description="useful for when you need to answer questions about AMD company financial statements"
    ),
    Tool(
        name="ASML Company Data",
        func=asml_agent.run,
        description="useful for when you need to answer questions about ASML company financial statements"
    ),
    Tool(
        name="AVGO Company Data",
        func=avgo_agent.run,
        description="useful for when you need to answer questions about AVGO company financial statements"
    ),
    Tool(
        name="Intel Company Data",
        func=intc_agent.run,
        description="useful for when you need to answer questions about Intel company financial statements"
    ),
    Tool(
        name="LRCX Company Data",
        func=lrcx_agent.run,
        description="useful for when you need to answer questions about LCRX company financial statements"
    ),
    Tool(
        name="MU Company Data",
        func=mu_agent.run,
        description="useful for when you need to answer questions about MU company financial statements"
    ),
    Tool(
        name="Nvidia Company Data",
        func=nvda_agent.run,
        description="useful for when you need to answer questions about Nvidia company financial statements"
    ),
    Tool(
        name="QCOM Company Data",
        func=qcom_agent.run,
        description="useful for when you need to answer questions about QCOM company financial statements"
    ),
    Tool(
        name="SSNLF Company Data",
        func=ssnlf_agent.run,
        description="useful for when you need to answer questions about SSNLF company financial statements"
    ),
    Tool(
        name="TSMC Company Data",
        func=tsm_agent.run,
        description="useful for when you need to answer questions about TSMC company financial statements"
    ),
    Tool(
        name="TXN Company Data",
        func=txn_agent.run,
        description="useful for when you need to answer questions about TXN company financial statements"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

llm=OpenAI(temperature=0)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

agent_chain.run("Compare Intel with its competitors financially.")