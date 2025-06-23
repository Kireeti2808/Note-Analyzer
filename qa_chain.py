from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def ask_question(retriever, query):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
