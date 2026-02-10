from openai import OpenAI
from dotenv import load_dotenv
import os

# 환경변수 로드 (.env)
load_dotenv()

# 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API 호출 테스트
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)