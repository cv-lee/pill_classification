# Print Formatting
multi_print_format ='shape loss: %.4f, color1 loss: %.4f, color2 loss: %.4f, total loss: %.4f, shape acc: %.4f, color1 acc: %.4f, color2 acc: %.4f, total acc: %.4f'

# Multi-label Model Loss Weights
shape_loss_weight = 1.0
color1_loss_weight = 1.0
color2_loss_weight = 1.0


# SHAPE INIT
s0= '삼각형'
s1 = '사각형'
s2 = '오각형'
s3 = '육각형'
s4 = '팔각형'
s5 = '원형'
s6 = '반원형'
s7 = '타원형'
s8 = '장방형'
s9 = '마름모형'
s10 = '기타'


# COLOR INIT
c0 = '빨강'
c1 = '주황'
c2 = '노랑'
c3 = '연두'
c4 = '초록'
c5 = '청록'
c6 = '파랑'
c7 = '남색'
c8 = '보라'
c9 = '분홍'
c10 = '자주'
c11 = '갈색'
c12 = '회색'
c13 = '검정'
c14 = '하양'
c15 = '투명'



def shapeConvert(shape):
    if shape == s0:
        return 0
    elif shape == s1:
        return 1
    elif shape == s2:
        return 2
    elif shape == s3:
        return 3
    elif shape == s4:
        return 4
    elif shape == s5:
        return 5
    elif shape == s6:
        return 6
    elif shape == s7:
        return 7
    elif shape == s8:
        return 8
    elif shape == s9:
        return 9
    elif shape == s10:
        return 10
    else:
        raise ValueError('Invalid shape input')

def colorConvert(color):
    if color == c0:
        return 0
    elif color == c1:
        return 1
    elif color == c2:
        return 2
    elif color == c3:
        return 3
    elif color == c4:
        return 4
    elif color == c5:
        return 5
    elif color == c6:
        return 6
    elif color == c7:
        return 7
    elif color == c8:
        return 8
    elif color == c9:
        return 9
    elif color == c10:
        return 10
    elif color == c11:
        return 11
    elif color == c12:
        return 12
    elif color == c13:
        return 13
    elif color == c14:
        return 14
    elif color == c15:
        return 15
    else:
        raise ValueError('Invalid color input')
