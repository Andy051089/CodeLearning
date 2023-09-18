age = int(input("請輸入你的年齡:"))
if 0< age < 18:
    print("你未成年")
elif age < 0 or age > 150:
    print("你不是人")
else:
    print("你成年啦")