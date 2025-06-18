# compare_vectors.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# 벡터 로드
place_vectors = np.load("vector_db/place_vectors.npy")
user_vector = np.load("vector_db/user_vector.npy").reshape(1, -1)

# 유사도 계산
similarities = cosine_similarity(user_vector, place_vectors)[0]
# 유사도 1.0인 항목 제외하고 상위 5개 추출
top_n = 5
filtered_indices = [i for i in similarities.argsort()[::-1] if similarities[i] < 1.0]
#top_indices = similarities.argsort()[::-1][:top_n]
top_indices = filtered_indices[:top_n]

# Top index 저장
with open("vector_db/top_place_indices.json", "w") as f:
     json.dump([int(i) for i in top_indices], f)

# 캡션 로드
with open("vector_db/place_info.json", "r", encoding="utf-8") as f:
    place_info = json.load(f)

print("📌 사용자 이미지와 유사한 여행지 Top {}:".format(len(top_indices)))
for rank, idx in enumerate(top_indices, start=1):
    image_name = place_info[idx]["name"]
    caption = place_info[idx]["caption"]
    similarity = similarities[idx]
    print(f"{rank}. {image_name} (유사도: {similarity:.4f})")
    print(f"   - 캡션: {caption}")
