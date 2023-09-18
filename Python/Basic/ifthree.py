num = int(input("請輸入一個數字:"))
if (num % 3 == 0 or num % 7 == 0) and (num % 21 != 0):
    print("此數可以被3或7整除，但不能被3和7整除")