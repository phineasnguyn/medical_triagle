import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# --- THAY ĐỔI Ở ĐÂY: Dùng Groq thay vì OpenAI ---
from langchain_groq import ChatGroq

# --- CẤU HÌNH API GROQ ---
# Thay thế bằng API Key bạn vừa lấy từ Groq
os.environ["GROQ_API_KEY"] = "gsk_Cjq34zbJWvUhBtgkIE54WGdyb3FYlPLnLQ6j4g08c1GPugsugLILE"

# Khởi tạo mô hình Llama 3 chạy trên hạ tầng siêu tốc của Groq
llm = ChatGroq(
    model="llama3-8b-8192", 
    temperature=0
)

# --- ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ĐẦU RA (JSON) ---
class PatientTriage(BaseModel):
    symptoms: str = Field(description="Các triệu chứng chính mà bệnh nhân đang gặp phải")
    duration: str = Field(description="Thời gian bắt đầu xuất hiện triệu chứng")
    severity: str = Field(description="Mức độ nghiêm trọng của triệu chứng")
    needs_immediate_care: bool = Field(description="Bệnh nhân có cần cấp cứu ngay lập tức không? (True/False)")

# Ép LLM trả về đúng định dạng
structured_llm = llm.with_structured_output(PatientTriage)

# --- THIẾT KẾ PROMPT ---
system_prompt = """
Bạn là một trợ lý y tế ảo thông minh. Nhiệm vụ của bạn là đọc lời khai của bệnh nhân bằng Tiếng Việt và trích xuất thông tin.
Tuyệt đối không chẩn đoán. Phải trả về dữ liệu bằng Tiếng Việt.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Lời khai của bệnh nhân: {patient_input}")
])

# --- HÀM XỬ LÝ ---
def extract_medical_info(patient_input: str):
    print("AI (Llama 3 via Groq) đang phân tích lời khai...\n")
    chain = prompt_template | structured_llm
    result = chain.invoke({"patient_input": patient_input})
    return result

if __name__ == "__main__":
    sample_input = "Chào bác sĩ, từ tối hôm qua tôi bị đau thắt ở vùng ngực trái, mồ hôi vã ra như tắm và cảm thấy rất khó thở. Tôi không chịu nổi nữa."
    
    extracted_data = extract_medical_info(sample_input)
    
    print("=== DỮ LIỆU ĐÃ TRÍCH XUẤT CHO BÁC SĨ (JSON) ===")
    print(extracted_data.model_dump_json(indent=2))