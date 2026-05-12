import pymupdf

doc = pymupdf.open("data.pdf")
full_text = ""

# 모든 페이지의 텍스트를 하나로 합치기
for page in doc:
    full_text += page.get_text()

# 'output.txt'라는 이름으로 저장 (인코딩은 utf-8 권장)
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

doc.close()
print("저장이 완료되었습니다.")