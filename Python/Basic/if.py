age = int(input("請輸入你的年齡:"))
if age >= 18:
    password = int(input("請輸入密碼"))
    if password == 123:
        print("歡迎光臨")
    else:
        print("請離開")
else:
    print("你未成年")