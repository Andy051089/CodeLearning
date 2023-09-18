import random

start = input('你要開始遊戲嗎? Yes or No ')
if start != 'yes':
    quit()
else:
    print('那我們猜數字遊戲開始... ')
long = input('請輸入你想的最大數: ')
if long.isdigit():
    long = int(long)
    if long <= 0:
        print('請輸入大於0的數字 ')
        quit()
else:
    print('請輸入數字 ')
    quit()
ans = random.randint(0, long)
count = 0
while True:
    count+=1
    guess = input('請猜: ')
    if guess.isdigit():
        guess=int(guess)
    else:
        print('請輸入一個數字')
        continue
    if guess != ans:
        if ans > guess:
            print(f'已經猜了{count}次，{guess}再大一點歐! ')
        else:
            print(f'已經猜了{count}次，{guess}再小一點歐! ')
    else:
        print(f'恭喜你，用{count}次猜中了，正確答案是{ans}')
        break