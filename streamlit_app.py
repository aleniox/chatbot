import streamlit as st
from PIL import Image
import datetime
from langgraph.graph import END, StateGraph
from prompt_template import *
from langchain_groq import ChatGroq

dt = datetime.datetime.now()
formatted = dt.strftime("%B %d, %Y %I:%M:%S %p")
image_human = Image.open("avata/avata_bot.png")
image_bot = Image.open("avata/avata_human.png")

llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
question_router = router_prompt | llm | JsonOutputParser()
generate_chain = generate_prompt | llm | StrOutputParser()
query_chain = query_prompt | llm | JsonOutputParser()
remind_chain = remind_prompt | llm | StrOutputParser()


def Agent():
    workflow = StateGraph(State)
    workflow.add_node("websearch", web_search)
    workflow.add_node("transform_query", transform_query)
    # workflow.add_node("docs_search", doc_search)
    workflow.add_node("generate", generate)

    # Build the edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "transform_query",
            "generate": "generate",
            # "docsearch": "docs_search",
        },
    )
    workflow.add_edge("transform_query", "websearch")
    workflow.add_edge("websearch", "generate")
    # workflow.add_edge("docs_search", "generate")
    workflow.add_edge("generate", END)

    # Compile the workflow
    local_agent = workflow.compile()
    return local_agent

def doc_search(state):
    question = state['question']
    if retriever:
        docs = retriever.invoke(question)
        print(docs)
    else:
        return {"context": None}
    search_result = ""
    for doc in docs:
        search_result += doc.page_content + "\n"
    with st.sidebar:
        # st.subheader("Search")
        st.write(docs)
    print("Search result:", search_result)
    return {"context": search_result}

def transform_query(state):
    print("Step: Tối ưu câu hỏi của người dùng")
    question = state['question']
    try:
        gen_query = query_chain.invoke({"question": question})
        search_query = gen_query["query"]
    except:
        search_query = question
    return {"search_query": search_query}

def web_search(state):
    search_query = state['search_query']
    print(f'Step: Đang tìm kiếm web cho: "{search_query}"')
    
    # Web search tool call
    search_result = web_search_tool.invoke(search_query)

    print("Search result:", search_result)
    return {"context": search_result}

def route_question(state):
    print("Step: Routing Query")
    question = state['question']
    try:
        output = question_router.invoke({"question": question})
        print('Lựa chọn của AI là: ', output)
    except:
        return "generate"
    if output['choice'] == "web_search":
        return "websearch"
    elif output['choice'] == 'generate':
        return "generate"
    # elif output['choice'] == 'docs_search':
    #     return "docsearch"

def generate(state):    
    print("Step: Đang tạo câu trả lời từ những gì tìm được")
    question = state["question"]
    context = state["context"]
    return {'question': question, 'context': context}

def plan_in_day():
    for chunk in remind_chain.stream({"time": formatted}):
        print(chunk, end="", flush=True)
        # st.session_state["full_message"] += chunk
        yield chunk

# def createdbformPdf(uploaded_files):
#     with st.spinner("Đang sử lý data..."):   
#         vectorstore = None     
#         if uploaded_files:
#             all_loader = []
#             for file in uploaded_files:
#                 with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                     temp_file.write(file.read())
#                     temp_file_path = temp_file.name
#                     pdf_loader = PyPDFLoader(temp_file_path)
#                     all_loader.extend(pdf_loader.load())
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#             chunked_documents = text_splitter.split_documents(all_loader)
#             vectorstore = FAISS.from_documents(chunked_documents, embedding_model)
#             retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#     return retriever 

def generate_response(prompt):
    local_agent = Agent()
    output = local_agent.invoke({"question": prompt})
    context = output['context']
    questions = output['question']
    with st.sidebar:
        st.header("🔎 Search")
        st.markdown(context)
    for chunk in generate_chain.stream({"context": context, "question": questions, "chat_history": chat_history}):
        print(chunk, end="", flush=True)
        st.session_state["full_message"] += chunk
        yield chunk
    chat_history.append(HumanMessage(content=questions))
    chat_history.append(AIMessage(content=st.session_state["full_message"]))
    with open('data/data_brain.pkl', 'wb') as fp:
        print(".............lưu data..........")
        pickle.dump(chat_history, fp)

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":speech_balloon:")
    st.title("💬 Chatbot")
        # with st.sidebar:
        # uploaded_files = st.file_uploader(
        # "Tải file các PDF của bạn ở đây", accept_multiple_files=True
        # )
        # if st.button("Process"):
        #     retriever = createdbformPdf(uploaded_files)
        #     print("Đây là tài liệu được gửi lên", retriever)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        with st.spinner("Đang lên kế hoạch..."):        
            st.write_stream(plan_in_day)
        # {"role": "assistant", "content": "Anh cần tôi giúp gì nào"}]

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar=image_human).write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar=image_bot).write(msg["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=image_human).write(prompt)
        st.session_state["full_message"] = ""
        with st.spinner("Đang tạo câu trả lời..."):        
            st.chat_message("assistant", avatar=image_bot).write_stream(generate_response(prompt))
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})   


if __name__ == "__main__":
    main()
