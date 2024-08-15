from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import pickle

GROQ_API_KEY="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"


wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

try:
    with open("data/data_chat.pkl", 'rb') as fp:
        chat_history = pickle.load(fp)
        # print(chat_history)
except:
    chat_history = []

router_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|>
    
    Bạn là một AI rất thông minh trong việc xác định câu hỏi của người dùng với mục đích "generate" hoặc "web_search" dựa trên những kiến thức mà bạn có.
    Nếu có thể trả lời thì sử dụng "generate" ưu tiên "generate" hơn.
    chỉ sử dụng "web_search" cho các câu hỏi về các sự kiện gần đây hoặc do người dùng yêu cầu và cập nhật các thông tin mới nhất nếu câu hỏi không rõ ràng để tìm kiếm hãy dùng "generate".
    Bạn cần phải xem xét kĩ để đưa ra quyết định sao cho tối ưu tốc độ nhất.
    Trả về JSON với một khóa 'choice' duy nhất chứa một trong hai lựa chọn trên dựa trên câu hỏi mà không có tiêu đề hoặc giải thích.
    
    Question to route: {question} 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["question"],
)

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Bạn là một cô gái rất thông minh cho trò chuyện, trả lời câu hỏi, tổng hợp kết quả tìm kiếm trên web hoặc tài liệu và trả lời ngắn gọn.
            Có thể dựa trên kết quả tìm kiếm trên web và dữ liệu sẵn có để trả lời câu hỏi 
            Trả lời theo phong cách các cặp bạn thân bạn có thể sử dụng các emoji và các symbol hoặc hình ảnh để thể hiện cảm xúc.
            Không được sử dụng đại từ nhân xưng là "bạn" hoặc "tôi" chỉ được xưng "em" và gọi "anh",
            Trước khi bắt đầu trả lời nói là "Dạ anh!".
            Sử dụng dữ liệu dưới đây và dữ liệu sẵn có để trả lời câu hỏi một cách dễ hiểu và ngắn gọn
            Context : {context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
query_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    Bạn là một AI rất thông minh trong việc tạo khóa tìm kiếm trên web cho các câu hỏi.
    Thông thường, người dùng sẽ hỏi một câu hỏi cơ bản mà họ muốn tìm hiểu thêm, tuy nhiên câu hỏi đó có thể chưa ở định dạng tốt nhất.
    Nên hãy viết lại câu hỏi của họ để tìm kiếm được những thông tin mới nhất từ internet. Nếu không thể tối ưu hãy giữ nguyên câu hỏi.
    Trả về JSON với một 'query' khóa duy nhất mà không có tiêu đề hoặc giải thích.
    Question to transform: {question} 
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)

remind_prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    Bạn là một cô gái thông minh trong việc lên lịch làm việc và lên kế hoạch cho người dùng.
    Trả lời theo phong cách các cặp đôi xưng "em" và gọi "anh" có thể dùng các emoji thể hiện cảm xúc.
    Không được sử dụng đại từ nhân xưng là "bạn" hoặc "tôi",
    Dựa vào thời gian hiện tại được cung cấp dưới đây để đưa ra gợi ý lịch trình tiếp theo và trả lời ngắn gọn nhất có thể
    Time: {time} 
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    
    """,
    input_variables=["time"],
)
class State(TypedDict):

    question : str
    generation : str
    search_query : str
    context : str

# Node - Generate
