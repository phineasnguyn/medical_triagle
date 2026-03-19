import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, List

# Nhập hàm speak_text từ file text_to_speech.py
from text_to_speech import speak_text

load_dotenv()

# ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU
class PatientTriage(BaseModel):
    raw_symptoms: str = Field(description="Lời khai gốc của bệnh nhân")
    standardized_symptoms: List[str] = Field(description="Danh sách các thuật ngữ y khoa chuẩn hóa")
    triage_level: Literal[
        "Level 1: Cấp cứu hồi sức (Nguy kịch)",
        "Level 2: Cấp cứu (Nguy cơ đe dọa tính mạng)",
        "Level 3: Khẩn cấp (Cần can thiệp sớm)",
        "Level 4: Bán khẩn cấp (Có thể chờ)",
        "Level 5: Không khẩn cấp (Khám thông thường)"
    ] = Field(description="Phân loại mức độ ưu tiên cấp cứu")
    recommended_department: str = Field(description="Khoa khám phù hợp")
    insurance_advice: str = Field(description="Hướng dẫn quyền lợi BHYT")
    required_documents: List[str] = Field(description="Danh sách giấy tờ cần mang")
    outpatient_workflow: List[str] = Field(description="Các bước quy trình khám")
    pre_exam_checklist: List[str] = Field(description="Checklist trước khi đi khám")
    emergency_red_flags: List[str] = Field(description="Cảnh báo dấu hiệu nguy hiểm")
    is_emergency: bool = Field(description="Đây có phải tình trạng nguy kịch không?")

# KHỞI TẠO GEMINI VÀ PROMPT
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Không tìm thấy GOOGLE_API_KEY. Vui lòng kiểm tra lại file .env!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=api_key
)

structured_llm = llm.with_structured_output(PatientTriage)

system_prompt = """
Bạn là Trợ lý Y tế Thông minh điều phối quy trình khám chữa bệnh.
Nhiệm vụ của bạn là phân tích lời khai và cung cấp lộ trình đi khám chi tiết.

TUÂN THỦ CÁC NGUYÊN TẮC:
1. KHÔNG CHẨN ĐOÁN: Không bao giờ được khẳng định bệnh nhân bị bệnh cụ thể gì.
2. ĐIỀU HƯỚNG: Gợi ý khoa khám phù hợp.
3. QUY TRÌNH (Outpatient): Hướng dẫn từ lúc Đăng ký -> Đóng phí -> Khám -> Xét nghiệm -> Lấy thuốc.
4. BHYT: Nhắc nhở về thẻ BHYT.
5. RED FLAGS: Cảnh báo cấp cứu nếu có dấu hiệu nguy hiểm.
6. CHECKLIST: Nhắc bệnh nhân những lưu ý trước khi khám.

Ngôn ngữ trả về: Tiếng Việt, lịch sự, rõ ràng.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Lời khai của bệnh nhân: {patient_input}")
])


def extract_medical_info(patient_input: str) -> PatientTriage:
    print(" AI đang phân tích lời khai...\n")
    chain = prompt_template | structured_llm
    return chain.invoke({"patient_input": patient_input})

def generate_spoken_response(data: PatientTriage) -> str:
    """Chuyển đổi dữ liệu từ AI thành văn bản tự nhiên để máy đọc"""
    if data.is_emergency:
        return (f"Cảnh báo! Tình trạng của bạn thuộc {data.triage_level}. "
                f"Với các dấu hiệu như {', '.join(data.emergency_red_flags)}, bạn cần đến khoa Cấp cứu ngay lập tức.")

    return (f"Hệ thống phân loại tình trạng của bạn là {data.triage_level}. "
            f"Bạn nên đến {data.recommended_department}. "
            f"Lưu ý trước khi đi: {', '.join(data.pre_exam_checklist)}. "
            f"Vui lòng mang theo {', '.join(data.required_documents)} để làm thủ tục.")


if __name__ == "__main__":
    print("=== TRỢ LÝ Y TẾ ẢO ĐÃ SẴN SÀNG ===")
    speak_text("Xin chào! Tôi là trợ lý y tế ảo của bạn. Hãy mô tả triệu chứng của bạn để tôi có thể hướng dẫn bạn quy trình khám phù hợp nhé.")
    print("Gõ 'thoát' để dừng chương trình.\n")
    
    while True:
        user_input = input(" Bệnh nhân: ")
        
        if user_input.strip().lower() in ['thoát','tạm biệt','bye', 'exit', 'quit']:
            print("Tạm biệt!")
            speak_text("Tạm biệt. Chúc bạn nhiều sức khỏe.")
            break
            
        try:
            # Bước 1: Gọi AI lấy JSON
            extracted_data = extract_medical_info(user_input)
            
            # (Tùy chọn) In JSON ra màn hình để kiểm tra
            print("\n[DỮ LIỆU ĐÃ BÓC TÁCH CHO BÁC SĨ (JSON)]")
            print(extracted_data.model_dump_json(indent=2, ensure_ascii=False))
            print("-" * 40)
            
            # Bước 2: Tạo kịch bản text để đọc
            spoken_text = generate_spoken_response(extracted_data)
            print(f"\n Trợ lý: {spoken_text}\n")
            
            # Bước 3: Đưa text vào Text-to-Speech
            speak_text(spoken_text)
            
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}\n")
            speak_text("Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.")