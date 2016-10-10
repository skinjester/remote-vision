def foo(bar):
    bar = 'new value'
    print(bar)

answer_list = 'old value'
foo(answer_list)
print(answer_list)
