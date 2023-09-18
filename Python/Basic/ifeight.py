num1 = int(input("請輸入第一個數:"))
num2 = int(input("請輸入第二個數:"))
if num1 >= num2:
    num3 = num1 - num2
    if num3 % 2 != 0:
        print(num3)
    else:
        print("這是偶數")
else:
    num4 = num2 - num1
    if num4 % 2 != 0:
        print(num4)
    else:
        print("這是偶數")
