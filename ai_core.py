import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, List

load_dotenv()

# 1. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ĐẦU RA CHO BÁC SĨ
class PatientTriage(BaseModel):
    raw_symptoms: str = Field(description="Lời khai gốc của bệnh nhân")
    standardized_symptoms: List[str] = Field(
        description="Danh sách các thuật ngữ y khoa chuẩn hóa được trích xuất từ lời khai (VD: 'đau bụng quá' -> 'Đau vùng bụng')"
    )
    triage_level: Literal[
        "Level 1: Cấp cứu hồi sức (Nguy kịch)",
        "Level 2: Cấp cứu (Nguy cơ đe dọa tính mạng)",
        "Level 3: Khẩn cấp (Cần can thiệp sớm)",
        "Level 4: Bán khẩn cấp (Có thể chờ)",
        "Level 5: Không khẩn cấp (Khám thông thường)"
    ] = Field(description="Phân loại mức độ ưu tiên cấp cứu")
    recommended_department: str = Field(description="Khoa khám phù hợp ( Khoa Nội, Khoa Ngoại, Khoa Sản, Khoa Nhi,...)")
    insurance_advice: str = Field(description="Hướng dẫn quyền lợi BHYT cơ bản cho trường hợp này")
    required_documents: List[str] = Field(description="Danh sách giấy tờ cần mang theo (CCCD, thẻ BHYT, sổ khám cũ...)")
    outpatient_workflow: List[str] = Field(description="Các bước quy trình khám ngoại trú cụ thể")
    pre_exam_checklist: List[str] = Field(description="Checklist trước khi đi khám (ví dụ: nhịn ăn, không uống cafe...)")
    emergency_red_flags: List[str] = Field(description="Cảnh báo các dấu hiệu nguy hiểm cần đi cấp cứu ngay lập tức")
    is_emergency: bool = Field(description="Đây có phải tình trạng nguy kịch không?")

# 2. KHỞI TẠO LLM GEMINI
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Không tìm thấy GOOGLE_API_KEY. Vui lòng kiểm tra lại file .env!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=api_key
)

# Với Gemini trong Langchain, chỉ cần truyền Pydantic class vào là nó tự hiểu
structured_llm = llm.with_structured_output(PatientTriage)

# 3. THIẾT KẾ PROMPT
system_prompt = """
Bạn là Trợ lý Y tế Thông minh điều phối quy trình khám chữa bệnh.
Nhiệm vụ của bạn là phân tích lời khai và cung cấp lộ trình đi khám chi tiết.

TUÂN THỦ CÁC NGUYÊN TẮC:
1. KHÔNG CHẨN ĐOÁN: Không bao giờ được khẳng định bệnh nhân bị bệnh cụ thể gì.
2. ĐIỀU HƯỚNG: Gợi ý khoa khám phù hợp (Nội khoa, Ngoại khoa, Da liễu, v.v.).
3. QUY TRÌNH (Outpatient): Hướng dẫn từ lúc Đăng ký -> Đóng phí -> Khám -> Xét nghiệm -> Lấy thuốc.
4. BHYT: Nhắc nhở về thẻ BHYT và giấy chuyển tuyến nếu có để hưởng quyền lợi đúng tuyến.
5. RED FLAGS: Nếu có dấu hiệu như khó thở, đau ngực dữ dội, liệt nửa người, mất ý thức... phải yêu cầu đi CẤP CỨU ngay.
6. CHECKLIST: Nhắc bệnh nhân nhịn ăn nếu nghi ngờ cần xét nghiệm máu, mang theo toa thuốc cũ.

Ngôn ngữ trả về: Tiếng Việt, lịch sự, rõ ràng.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Lời khai của bệnh nhân: {patient_input}")
])

# 4. HÀM XỬ LÝ
def extract_medical_info(patient_input: str):
    print("AI (Gemini) đang phân tích lời khai...\n")
    chain = prompt_template | structured_llm
    result = chain.invoke({"patient_input": patient_input})
    return result

if __name__ == "__main__":
    sample_input = "Chào bác sĩ, từ tối hôm qua tôi bị đau thắt ở vùng ngực trái, mồ hôi vã ra như tắm và cảm thấy rất khó thở. Tôi không chịu nổi nữa."
    
    try:
        extracted_data = extract_medical_info(sample_input)
        print("=== DỮ LIỆU ĐÃ TRÍCH XUẤT CHO BÁC SĨ (JSON) ===")
        print(extracted_data.model_dump_json(indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        
if __name__ == "__main__":
    print("=== TRỢ LÝ Y TẾ ẢO ĐÃ SẴN SÀNG ===")
    print("Gõ 'thoát' để dừng chương trình.\n")
    
    while True:
        # Nhập lời khai trực tiếp từ Terminal
        user_input = input("👤 Bệnh nhân: ")
        
        # Điều kiện thoát
        if user_input.strip().lower() in ['thoát','tạm biệt','bye', 'exit', 'quit']:
            print("Tạm biệt!")
            break
            
        try:
            extracted_data = extract_medical_info(user_input)
            print("\n DỮ LIỆU ĐÃ BÓC TÁCH (JSON):")
            print(extracted_data.model_dump_json(indent=2, ensure_ascii=False))
            print("-" * 40 + "\n")
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}\n")