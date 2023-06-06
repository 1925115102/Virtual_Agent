
positionList = []
number = 0

def appendPosition(x):
    global number
    global positionList
    if number < 60:
        positionList.append(x)
        number += 1
    else:
        a = velocity()
        number = 0
        positionList = []
        return a


def velocity():
    return positionList[-1] - positionList[0]



