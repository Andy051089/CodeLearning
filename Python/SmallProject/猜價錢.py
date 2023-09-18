import random

price = random.randint(0, 100)
print(price)
time = 0
total = 5
while time < total:
    player_a = int(input('請輸入價格'))
    player_b = int(input('請輸入價格'))
    if player_a == price or player_b == price:
        break
    else:
        if abs(player_a - price) > abs(player_b - price):
            print('B玩家快對了')
        elif abs(player_a - price) < abs(player_b - price):
            print('A玩家快對了')
        elif abs(player_a - price) == abs(player_b - price):
            print('都很接近了')
    time += 1
print('遊戲結束')
if player_a == price:
    print('恭喜A玩家猜對了')
elif player_b == price:
    print('恭喜B玩家猜對了')
elif abs(player_a - price) > abs(player_b - price):
    print('B玩家贏摟')
else:
    print('A玩家贏了')
