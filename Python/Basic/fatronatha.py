# num   total
# 1     1      2 3 5 8
a = int(input("要幾個"))
num = 0
total = 1
print(num, total, end=" ")
for i in range(1, int(a-1)):
    a = num
    num = total
    total += a
    print(total, end=" ")
