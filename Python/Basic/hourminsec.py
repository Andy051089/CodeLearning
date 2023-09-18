total_second = 3718
hour = total_second // 3600
min = (total_second % 3600) // 60
#second = 15678 - ((3600 * hour) + (60 * min))
second = total_second % 60
print(hour, "小時", min, "分鐘", second, "秒", sep = "")


