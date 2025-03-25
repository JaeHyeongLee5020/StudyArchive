#거북이가 달려간 x,y,좌표를 화면에 표시
#선의 색상과 거북이 색상을 일치.
#실행시킬때 마다 100마리 이하의 거북이 수 설정.


import turtle
import random

turtleList = []
colorList = ["red" , "green", "blue", "black", "magenta","orange","gray"]

shapeList = ["arrow","circle","square","triangle","turtle"]
a = 0
turtle.setup(550,550)
turtle.screensize(500,500)
turtle.speed(1)

for i in range(0,100) :
    shape = random.choice(shapeList)
    color = random.choice(colorList)
    x = random.randint(-250,250)
    y = random.randint(-250,250)
    t = turtle.Turtle(shape)
    tup = (t, color, x ,y )
    
    turtleList.append(tup)

for tup in turtleList :
    a+=1
    t = tup[0]
    t.pencolor( tup[1] )
    t.goto(tup[2],tup[3])
    print("현재 위치 : x =",t.xcor(),", y =",t.ycor()) #idle 창에도 출력.
    t.write(f"현재위치 : x={x}, y={y} {a}번째 좌" ,align="right") #터틀 그래픽으로 오른쪽에 출력.

turtle.done()

