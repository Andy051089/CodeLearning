for num in range(101, 1000):
    for ans in range(2, num):
        if num % ans == 0:
            break
    else:
        print(num)