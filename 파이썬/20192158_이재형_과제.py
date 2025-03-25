import turtle

t = turtle.Turtle()

movement_history = []

def move_forward(distance):
    t.forward(distance)
    movement_history.append(('forward', distance))

def move_backward(distance):
    t.backward(distance)
    movement_history.append(('backward', distance))

def turn_left(angle):
    t.left(angle)
    movement_history.append(('left', angle))

def turn_right(angle):
    t.right(angle)
    movement_history.append(('right', angle))

while True:
    
    command = input("Enter command (forward, backward, left, right, exit): ").lower()

    if command == 'exit':
        break

    if command == 'forward' or command == 'backward':
        distance = int(input("Enter distance: "))
        if command == 'forward':
            move_forward(distance)
        else:
            move_backward(distance)
    elif command == 'left' or command == 'right':
        angle = int(input("Enter angle: "))
        if command == 'left':
            turn_left(angle)
        else:
            turn_right(angle)
    else:
        print("Invalid command. Please enter a valid command.")

turtle.bye()

with open('movement_history.txt', 'w') as file:
    for move in movement_history:
        file.write(f"{move[0]} {move[1]}\n")
