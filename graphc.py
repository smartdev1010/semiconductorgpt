import openai
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    df = pd.read_excel("./xlsxs/Merged data.xlsx")
    df = df.astype(str)
    xlsx_string = df.to_csv(index=False, sep="\t")

    query = input("Input your query: ")

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
    extract = code.split("```")[1]
    print(extract)

    exec(extract)
