import sys
from datetime import datetime
import numpy as np


now = datetime.now()
def print_custum(color, bg_color, msg):
    color_code = '\033['+color+'m\033['+bg_color+'m'
    print(color_code+msg+'\033[0m')

def print_black(msg, reset):
    print('\033[30m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None


def print_red(msg, reset):
    print('\033[31m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None

def print_green(msg, reset):
    print('\033[32m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None

def print_yellow(msg, reset):
    print('\033[33m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None

def print_blue(msg, reset):
    print('\033[34m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None

def print_magenta(msg, reset):
    print('\033[35m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None


def print_cyan(msg, reset):
    print('\033[36m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None

def print_white(msg, reset):
    print('\033[37m', end="")
    print(msg, end="")
    if reset == True:
        print('\033[0m')
    elif reset==False:
        None



def bg_print_red():
    print('\033[41m', end="")

def bg_print_green():
    print('\033[42m', end="")

def bg_print_yellow():
    print('\033[43m', end="")

def bg_print_blue():
    print('\033[44m', end="")

def bg_print_magenta():
    print('\033[45m', end="")

def bg_print_cyan():
    print('\033[46m', end="")

def bg_print_white():
    print('\033[47m', end="")
def reset(end=False):
    if end == False:
        print('\033[0m')
    if end == True:
        print('\033[0m', end="")
def br():
    print()



def bar_chart(args = [], name = []):
#     graph =
# 10|
# 9 |
# 8 |
# 7 |
# 6 |
# 5 |
# 4 |
# 3 |
# 2 |
# 1 |
# 0 |
#    - - -
#    0 1
    #true, true, false, false, false, false, false
    """_summary_

    Args:
        args (list, optional): _description_. Defaults to [].
        name (list, optional): _description_. Defaults to [].
    """
    def bar_color(args):
        color = [1,2,3,4,5,6,7]
        select_color  = []
        # print(args, end="")
        for a in range(len(args)):
            if args[a] == True:
                select_color.append(color[a%7])
            else:
                select_color.append(0)
        # print(select_color, end="")
        for b in range(len(select_color)):

            if select_color[b] == 0:
                print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 1:
                bg_print_blue();print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 2:
                bg_print_cyan();print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 3:
                bg_print_green();print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 4:
                bg_print_magenta();print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 5:
                bg_print_red();print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 6:
                bg_print_yellow();print("  ", end="");reset(True);print(" ", end="")
            elif select_color[b] == 7:
                bg_print_white();print("  ", end="");reset(True);print(" ", end="")
            # br()



    biggest = args[0]
    space=""
    for i in range(len(args)):
        if args[i] > biggest:
            biggest = args[i]
    alen = len(str(biggest))
    biggest_plus_five = biggest+5
    for height in range(biggest_plus_five):
        x = biggest-(height-3)
        bar = []
        for a in range(len(args)):
            if x<=args[a]:
                bar.append(True)
            elif x>args[a]:
                bar.append(False)

        for a in range(alen-len(str(x))):
            space+=" "
        if x >0:
            print(x, space, "|", end="");bar_color(bar);br()
        elif x==0:
            print(x, space, "|", end="");bar_color(bar);br()
            print(" ", space, "|", end="")
            for i in range(len(args)):
                print("__ ", end="")
            print()
        endspace = space
        space = ""
    print("  ", space, args, biggest)
    print()
    print("color, index, name")
    color = [1,2,3,4,5,6,7]
    select_color  = []
    for a in range(len(name)):
        select_color.append(color[a%7])
    for b in range(len(select_color)):
        if select_color[b] == 1:
            bg_print_blue();print("    ", end="");reset(True);print("  ", b, "    ",name[b])
        elif select_color[b] == 2:
            bg_print_cyan();print("    ", end="");reset(True);print("  ", b, "    ",name[b])
        elif select_color[b] == 3:
            bg_print_green();print("    ", end="");reset(True);print("  ", b, "    ",name[b])
        elif select_color[b] == 4:
            bg_print_magenta();print("    ", end="");reset(True);print("  ", b, "    ",name[b])
        elif select_color[b] == 5:
            bg_print_red();print("    ", end="");reset(True);print("  ", b, "    ",name[b])
        elif select_color[b] == 6:
            bg_print_yellow();print("    ", end="");reset(True);print("  ", b, "    ",name[b])
        elif select_color[b] == 7:
            bg_print_white();print("    ", end="");reset(True);print("  ", b, "    ",name[b])


def band_graph(value = [], name = []):
    standard = 0
    for i in range(len(value)):
        standard += value[i]

    percent = []
    for i in range(len(value)):
        percent.append((value[i]*100)/standard)
