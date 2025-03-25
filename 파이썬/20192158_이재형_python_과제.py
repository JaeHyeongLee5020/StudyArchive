import random

computer, user = 0, 0

k = int(input("몇번의 기회를 드릴까요?"))
m = int(input("1부터 몇까지의 범위를 설정할까요?"))
count = 0

computer = random.randint (1, m)
print("10번의 기회가 있습니다. 범위는 1부터",m," 사이 입니다.")
for i in range(k, 1, -1) :
    user = int(input("컴퓨터가 생각한 숫자는 ? "))
    print(k-1,"번의 기회 남았습니다 :")
    count +=1
    if computer == user :
        break
    elif computer > user:
        print("컴퓨터가 생각한 수는 ",user,"보다 큽니다. 다시 도전하세요")
        continue
    else :
        print("컴퓨터가 생각한 수는 ",user,"보다 작습니다. 다시 도전하세요")
        continue

print(count,'번만에 맞췄습니다. 게임을 종료합니다.')
