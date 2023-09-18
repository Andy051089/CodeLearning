height = float(input("請輸入您的身高:"))
weight = float(input("請輸入你的體重:"))
BMI = weight / (height ** 2)
if BMI >= 18.5 and BMI <= 24.9:
    print(BMI, "+", "你是正常身材", sep ="")
elif BMI >= 24.9:
    print(BMI, "+", "你過重", sep = "")
else:
    print(BMI, "+", "你過輕", sep = "")