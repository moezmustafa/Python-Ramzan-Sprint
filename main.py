import os
import time  # Import the time module

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

def create_retrieval_qa_chain(llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

def load_model(
    model_path="model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    temperature=0.01,
    context_length=10024,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        temperature=temperature,
        context_length=context_length,
    )
    return llm

def create_retrieval_qa_bot(model_name="sentence-transformers/all-MiniLM-L6-v2", persist_dir="db", device="mps"):
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No directory found at {persist_dir}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    try:
        llm = load_model()
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    try:
        qa = create_retrieval_qa_chain(llm=llm, db=db)
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa

def retrieve_bot_answer(query):
    start_time = time.time()  # Record the start time
    qa_bot_instance = create_retrieval_qa_bot()
    bot_response = qa_bot_instance.invoke({"query": query})
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    bot_response['time_taken'] = elapsed_time  # Optionally add the time taken to the response
    return bot_response



# if __name__ == "__main__":
#     while True:
#         query = input("Enter your query (or type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             print("Exiting...")
#             break
#         response = retrieve_bot_answer(query)
#         print(response['result'])
#         print(f"Time taken: {response['time_taken']:.2f} seconds")
#         print('done')




import chainlit as cl

@cl.on_chat_start
async def initialize_bot():
   
    qa_chain = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using MISTRAL  and LangChain."
    )
    await welcome_message.update()

    cl.user_session.set("chain", qa_chain)


@cl.on_message
async def process_chat_message(message):
    qa_chain = cl.user_session.get("chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    response=await qa_chain.ainvoke(message.content, callbacks=[callback_handler])
    bot_answer = response["result"]
    source_documents = response["source_documents"]

    # if source_documents:
    #     bot_answer += f"\nSources:" + str(source_documents)
    # else:
    #     bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send()


