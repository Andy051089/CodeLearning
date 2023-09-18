a=['出去玩']
b=['逛街']
c=['睡覺']
out_in_door =input('你要 1.出去玩還是 2.在家')
if out_in_door in a:
    disneyland_street=input('那要1.逛街還是去 2.遊樂園')
    if disneyland_street in b:
        print('我要出去逛街')
    else:
        print('我要出去遊樂園')
else:
    sleep_video = input('那要1.睡覺還是去 2.看影片')
    if sleep_video in c:
        print('我要在家睡覺')
    else:
        print('我要在家看影片')