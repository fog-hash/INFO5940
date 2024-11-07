# adopted from https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/explainable_retrieval.ipynb

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from os import environ

class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")  # Adjust as needed
        self.vectorstore = Chroma.from_texts(texts, self.embeddings)

        # Create a base retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Create an explanation chain using Azure OpenAI
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Analyze the relationship between the following query and the retrieved context.
                        Explain why this context is relevant to the query and how it might help answer the query.
                        
                        Query: {query}
                        
                        Context: {context}
                        
                        Explanation:"""
        )

        # Use AzureChatOpenAI from LangChain
        self.llm = llm = AzureChatOpenAI(
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            api_key=environ['AZURE_OPENAI_API_KEY'],
            api_version="2023-03-15-preview",
            azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
        )


        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)

        explained_results = []

        for doc in docs:
            # Generate explanation
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content

            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })

        return explained_results


texts = [
    # "Cornell Diary has the best ice cream in the world because it has its own teaching farm.",
    # "People prefer organic milk verses regular milk because it is healthier.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    # "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]
 
explainable_retriever = ExplainableRetriever(texts)

import streamlit as st

st.title("Awesome Assignment 2 Chatbot")
st.caption("Powered by INFO-5940 Group 6")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": str(texts)}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input and process the question
question = st.chat_input("Ask questions based on the context above", disabled=False)

if question:
    # Append the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
            
    # Process the user's question using the RAG chain with retry logic
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            response = explainable_retriever.retrieve_and_explain(question)
            if response:
                for i, result in enumerate(response, 1):
                    st.write(f"""
                        Result {i}:  
                        **Content**: {result['content']}  
                        **Explanation**: {result['explanation']}
                    """)

    # Append the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})    

