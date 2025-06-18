import numpy as np
from extract_clip_vector import extract_image_vector
from blip_caption import generate_caption

image_path = "./user_input/lotteworld.jpg"
vector = extract_image_vector(image_path)
caption = generate_caption(image_path)

np.save("vector_db/user_vector.npy", vector)

with open("vector_db/user_caption.txt", "w", encoding="utf-8") as f:
    f.write(caption)

print("✅ 사용자 벡터 및 캡션 저장 완료")
print(f"📷 이미지 설명 [clip] : {caption}")
