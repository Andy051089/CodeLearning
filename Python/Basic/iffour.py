year = int(input("請輸入一個年份:"))
if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
    print(year,"此為閏年",sep = "")
else:
    print(year,"此為平年",sep = "")