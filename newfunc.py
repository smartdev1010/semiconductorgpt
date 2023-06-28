import openai
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# daf = pd.read_csv("./data.csv")
# print(daf)


# Prepare the prompt with CSV data and additional context
def prepare_prompt(csv_string, prompt):
    full_prompt = f"Here is the xlsx data:\n{csv_string}\n{prompt}\n\nAlways start writing your code with a unique marker like 'Code starts here:'\n"
    return full_prompt


def fetch_and_visualize_financial_data(keyword: str, timeframe: str, company: str):
    """Fetch appropriate CSV files in the database related to the given keyword and generate a visualization of the data trends within the given timeframe. Parameters: keyword (str): The financial keyword (e.g., 'revenue', 'costs'). timeframe (str): The timeframe for the data trend (e.g., 'last 5 years'). Returns: dict: Visualization data"""  #

    # Read the xlsx file into a dataframe
    # df = pd.read_excel("./xlsxs/intc_financial_statement.xlsx")
    # Convert all columns to strings
    # df = df.astype(str)
    # Convert dataframe to string format
    # xlsx_string = df.to_csv(index=False, sep="\t")

    full_prompt = prepare_prompt(
        "",
        f"Give me code in python to plot the analyze the {keyword} in the {timeframe}.",
    )

    print(full_prompt)

    # Generate completion using OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a McKinsey consultant analyzing financial information.",
            },
            {"role": "user", "content": full_prompt},
        ],
    )
    # Extract the generated code from the response
    print(response)
    generated_code = (
        response["choices"][0]["message"].content.split("Code starts here:")[-1].strip()
    )
    generated_code = generated_code.replace("python", "")
    generated_code = generated_code.replace(
        "data.xlsx", f"./xlsxs/{company}_financial_statement.xlsx"
    )

    # Execute the generated code
    exec(generated_code.split("```")[1])
    # # Display the plot
    # plt.show()


functions = [
    {
        "name": "fetch_and_visualize_financial_data",
        "description": "Fetch and visualize financial data trends",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Financial keyword, e.g., 'revenue', 'costs'",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for the data trend, e.g., 'last 5 years'",
                },
                "company": {
                    "type": "string",
                    "description": "Company Token, e.g., 'amat', 'amd', 'asml', 'avgo', 'intc', 'lrcx', 'mu', 'nvda, 'qcom', 'ssnlf', 'tsm', 'txn'",
                },
            },
            "required": ["keyword", "timeframe"],
        },
    }
]

question = "Compare the top 5 semiconductor companies financially over the last 3 years"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    temperature=0,
    messages=[
        {"role": "user", "content": question},
    ],
    functions=functions,
    function_call="auto",
)

message = response["choices"][0]["message"]

if message.get("function_call"):
    function_name = message["function_call"]["name"]
    arguments = json.loads(message["function_call"]["arguments"])

    # Call the appropriate function
    if function_name == "fetch_and_visualize_financial_data":
        function_response = fetch_and_visualize_financial_data(
            keyword=arguments.get("keyword"),
            timeframe=arguments.get("timeframe"),
            company=arguments.get("company"),
        )

    # second_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-16k",
    #     temperature=0,
    #     messages=[
    #         {"role": "user", "content": question},
    #         message,
    #         {
    #             "role": "function",
    #             "name": function_name,
    #             "content": function_response,
    #         },
    #     ],
    # )

    # print(second_response.choices[0]["message"]["content"].strip())

# else:
# print(response.choices[0]["message"]["content"].strip())
