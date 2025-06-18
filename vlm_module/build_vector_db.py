import os
import numpy as np
import json
from extract_clip_vector import extract_image_vector
from blip_caption import generate_caption

# 디렉토리 생성 코드 추가 (필요 시만 생성)
os.makedirs("../vector_db", exist_ok=True)

IMG_DIR = "./travel_images"
SAVE_VEC = "vector_db/place_vectors.npy"
SAVE_META = "vector_db/place_info.json"

vectors = []
meta_info = []

for fname in os.listdir(IMG_DIR):
    if fname.endswith(".jpg") or fname.endswith(".png"):
        path = os.path.join(IMG_DIR, fname)

        vec = extract_image_vector(path)
        caption = generate_caption(path)

        vectors.append(vec)
        meta_info.append({
            "filename": fname,
            "name": fname.replace(".jpg", "").replace(".png", "").replace("_", " ").title(),
            "caption": caption
        })

np.save(SAVE_VEC, np.array(vectors))
with open(SAVE_META, "w", encoding="utf-8") as f:
    json.dump(meta_info, f, ensure_ascii=False)

print("✅ 여행지 벡터 + 캡션 저장 완료")
