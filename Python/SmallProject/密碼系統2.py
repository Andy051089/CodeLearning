import random
import string

nums=string.digits
words=string.ascii_letters
password=list(nums+words)
random.shuffle(password)
how_much = int(input('請輸入要幾位數'))
final_password=''.join(password[: how_much])
print(final_password)
