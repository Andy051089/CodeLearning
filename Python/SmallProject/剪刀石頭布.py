import random
player = int(input("請入(1)剪刀 (2)石頭 (3)布"))
computer = random.randint(1, 3)
if (player == 1 and computer == 3) or (player == 2 and computer == 1) or (player == 3 and computer == 2):
    print("你贏了")
elif player == computer:
    print("平局")
else:
    print("你輸了")
