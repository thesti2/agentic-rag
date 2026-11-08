from dotenv import load_dotenv
from graph.graph import app

load_dotenv()



if __name__ == '__main__':
    print("hello RAG")
    result = app.invoke(input={"question": "what is agent memory?"})

    print(result)