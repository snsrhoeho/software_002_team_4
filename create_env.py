import os

# 저장 경로 설정
env_path = os.path.join(os.getcwd(), ".env")

# 사용자로부터 API 키 입력 받기
api_key = input("🔐 OpenAI API 키를 입력하세요 (sk-로 시작): ").strip()

# 기본 유효성 검사
if not api_key.startswith("sk-") or len(api_key) < 20:
    print("❌ 올바른 OpenAI API 키 형식이 아닙니다.")
    exit(1)

# .env 파일 생성
with open(env_path, "w", encoding="utf-8") as f:
    f.write(f"OPENAI_API_KEY={api_key}\n")

print(f"✅ .env 파일이 생성되었습니다: {env_path}")
