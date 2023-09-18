def add(num1, num2):
    return num1 + num2


def minus(num1, num2):
    return num1 - num2


def times(num1, num2):
    return num1 * num2


def divide(num1, num2):
    ans = num1 // num2
    left = num1 % num2
    return ans, left


while True:
    what_type = input('請輸入要做的運算(1)+ (2)- (3)* (4)/ 或是其他數字停止')
    if what_type in ('1', '2', '3', '4'):
        num1 = int(input('請輸入第一個數字'))
        num2 = int(input('請輸入第二個數字'))
        if what_type == '1':
            print(f'{num1}+{num2}={add(num1, num2)}')
        elif what_type == '2':
            print(f'{num1}-{num2}={minus(num1, num2)}')
        elif what_type == '3':
            print(f'{num1}*{num2}={times(num1, num2)}')
        elif what_type == '4':
            if num1 % num2 == 0:
                print(f'{num1}/{num2}={divide(num1, num2)[0]}')
            else:
                print(f'{num1}/{num2}={divide(num1, num2)[0]}...{divide(num1, num2)[1]}')
    else:
        break
