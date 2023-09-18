for num in range(100, 1000):  # 153
    num1 = num % 10
    num2 = (num // 10) % 10
    num3 = num // 100
    if (num1 ** 3) + (num2 ** 3) + (num3 ** 3) == num:
        print(num)
