names_scores = ['Andy', 98, 'Amber', 86, 'John', 88]
tools = input('(A)查詢(B)新增(C)刪除(D)修改').upper()
if tools == 'A':
    whose_score = input('你要查詢誰的成績').title()
    if whose_score not in names_scores:
        print('查無此人')
    else:
        name_score = names_scores.index(whose_score)
        print(f'{whose_score}的成績是{names_scores[name_score + 1]}分')
elif tools == 'B':
    new_name = input('請新增新的名子').title()
    if new_name in names_scores:
        print('已經有這個人')
    else:
        new_score = int(input('請新增成績'))
        names_scores.extend([new_name, new_score])
        print('Success')
        print(names_scores)
elif tools == 'C':
    dele = input('你要刪除誰').title()
    dele_name = names_scores.index(dele)
    names_scores.pop(dele_name)
    names_scores.pop(dele_name)
    print(f'已刪除成功{names_scores}')
elif tools == 'D':
    change = input('要修改誰的成績').title()
    new_score = int(input('新的成績'))
    score_change = names_scores.index(change)
    names_scores[score_change + 1] = new_score
    print(names_scores)
