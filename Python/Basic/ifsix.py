score = int(input("請輸入你的成績:"))
if score < 60:
    print(score, "你的成績不及格")
elif 60 <= score <= 80:
    print(score, "你的成績剛好及格")
elif 81 <= score <= 90:
    print(score, "你的成績良好")
else:
    print(score, "你的成績優秀")