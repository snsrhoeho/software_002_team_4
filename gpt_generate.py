import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# .env에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 사용자 캡션 로드
with open("vector_db/user_caption.txt", "r", encoding="utf-8") as f:
    user_caption = f.read().strip()

# 장소 정보 로드 (caption + name)
with open("vector_db/place_info.json", "r", encoding="utf-8") as f:
    place_info = json.load(f)

# 벡터 로딩 및 유사도 계산
place_vectors = np.load("vector_db/place_vectors.npy")
user_vector = np.load("vector_db/user_vector.npy").reshape(1, -1)
similarities = cosine_similarity(user_vector, place_vectors)[0]

# Top 5 index 불러오기 (compare_vectors.py에서 저장한 결과)
with open("vector_db/top_place_indices.json", "r") as f:
    top_indices = json.load(f)

# 추천 결과 정리
top_places = [
    {
        "name": place_info[i]["name"],
        "caption": place_info[i]["caption"],
        "score": round(similarities[i], 4)
    }
    for i in top_indices
]

# 콘솔 출력 - 기본 정보
print(f"\n📌 사용자 이미지 설명: {user_caption}\n")
print("📍 직접 선정한 추천 여행지 TOP 5:")
for i, place in enumerate(top_places, 1):
    print(f"{i}. {place['name']} (유사도: {place['score']:.4f})")

# GPT-4o에 전송할 메시지 구성
place_caption_list = [f"- {p['name']}: {p['caption']}" for p in top_places]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "당신은 감성적인 여행 플래너입니다. 사용자에게 꼭 맞는 장소를 따뜻한 문장으로 추천해주세요."},
        {"role": "user", "content":
            f"사용자 이미지 설명: '{user_caption}'\n"
            f"다음은 유사한 장소들입니다:\n" +
            "\n".join(place_caption_list) +
            "\n각 장소의 분위기나 특징을 고려해서, 사용자에게 가장 잘 맞는 장소 5곳을 추천하고 이유도 설명해주세요."
        }
    ],
    temperature=0.7,
)

# GPT 응답 출력
print("\n📝 오직 GPT만의 장소 추천:\n")
print(response.choices[0].message.content.strip())
