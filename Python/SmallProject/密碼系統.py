password = '1234'
for i in range(0, 5):
    password_input = input('請輸入密碼')
    if password_input == password:
        print('歡迎光臨')
        break
    elif password_input != password and i < 4 :
        print(f'你輸入的密碼錯誤，還剩下{4 - i}次機會')
    else:
        print('已經輸入錯誤太多次了')
