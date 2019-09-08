# libraries
from libraries import *

# local libraries
from parameters import *
from bound_maker import *
from Custom_layers import *
from functions import *

def boundary_maker():
    dt = np.load("D://IMGRESULTS/ARRAYS/data.npy")
    sc = np.load("D://IMGRESULTS/ARRAYS/score.npy")
    rs = np.load("D://IMGRESULTS/ARRAYS/result.npy")

    dt = dt[img]
    sc = sc[img]
    sc = sc.reshape((OUTPUT_SIZE, OUTPUT_SIZE))
    rs = rs[img]
    IMG = Image.fromarray(dt.astype('uint8'), 'RGB')
    IMG.show()
    IMG = Image.fromarray(COEF * sc.astype('uint8'), 'L')
    IMG.show()
    IMG = Image.fromarray(COEF * rs.astype('uint8'), 'L')
    IMG.show()

    hit_map = np.ones((OUTPUT_SIZE, OUTPUT_SIZE))
    bound_map = np.zeros((INPUT_SIZE, INPUT_SIZE))
    stop = False

    for row in range(hit_map.shape[0]):
        for pil in range(hit_map.shape[1]):
            if stop == True:
                continue
            if hit_map[row, pil] == 0:
                continue

            if rs[row, pil] == 0:
                hit_map[row, pil] = 0
                continue

            left = pil
            right = pil
            top = row
            bot = row
            count = 0

            hit_map[row, pil] = 0

            list = [[row, pil]]

            while len(list) != 0:
                #print(list)
                #print("COUNT ", count)
                #print("LEFT", left)
                #print("RIGHT ", right)
                #print("TOP ", top)
                #print("BOT ", bot, "\n")
                count += 1
                list1 = []
                for ls in list:
                    i = ls[0]
                    j = ls[1]


                    # QUARTER 1
                    if j + 1 < OUTPUT_SIZE:
                        if hit_map[i, j + 1] == 1 and rs[i, j + 1] == 1:
                            hit_map[i, j + 1] = 0
                            list1.append([i, j + 1])

                            if j + 1 > right:
                                right = j + 1

                        if hit_map[i, j + 1] == 0 or rs[i, j + 1] == 0:
                            hit_map[i, j + 1] = 0

                    # QUARTER 2
                    if i - 1 > 0:
                        if hit_map[i - 1, j] == 1 and rs[i - 1, j] == 1:
                            hit_map[i - 1, j] = 0
                            list1.append([i - 1, j])

                            if i - 1 < top:
                                top = i - 1

                        if hit_map[i - 1, j] == 0 or rs[i - 1, j] == 0:
                            hit_map[i - 1, j] = 0

                    # QUARTER 3
                    if j - 1 > 0:
                        if hit_map[i, j - 1] == 1 and rs[i, j - 1] == 1:
                            hit_map[i, j - 1] = 0
                            list1.append([i, j - 1])

                            if j - 1 < left:
                                left = j - 1

                        if hit_map[i, j - 1] == 0 or rs[i, j - 1] == 0:
                            hit_map[i, j - 1] = 0

                    # QUARTER 4
                    if i + 1 < OUTPUT_SIZE:
                        if hit_map[i + 1, j] == 1 and rs[i + 1, j] == 1:
                            hit_map[i + 1, j] = 0
                            list1.append([i + 1, j])

                            if i + 1 > bot:
                                bot = i + 1

                        if hit_map[i + 1, j] == 0 or rs[i + 1, j] == 0:
                            hit_map[i + 1, j] = 0

                list = list1
            #print("LEFT", left)
            #print("RIGHT ", right)
            #print("TOP ", top)
            #print("BOT ", bot)
            if line_thickness <= 1:
                print(" T ", top)
                print(" B ", bot)
                print(" L ", left)
                print(" R ", right)
                real_top = max(top * FACTOR - FACTOR + 1, 0)
                real_bot = min(bot * FACTOR + 1, INPUT_SIZE)
                real_left = max(left * FACTOR - FACTOR + 1, 0)
                real_right = min(right * FACTOR + 1, INPUT_SIZE)


                d_vert = math.floor((math.ceil(real_bot - real_top) / 2) / 0.5)
                d_hor = math.floor((math.ceil(real_right - real_left) / 2) / 0.4)

                real_top = max(real_top - d_vert, 0)
                real_bot = min(real_bot + d_vert, INPUT_SIZE - 1)
                real_left = max(real_left - d_hor, 0)
                real_right = min(real_right + d_hor, INPUT_SIZE - 1)

                print(real_top)
                print(real_bot)
                print(real_left)
                print(real_right)

                for i in range(real_top, real_bot):
                    bound_map[i, real_left] = 1
                    bound_map[i, real_right] = 1
                for j in range(real_left, real_right):
                    bound_map[real_top, j] = 1
                    bound_map[real_bot, j] = 1

            if line_thickness > 1:
                it = math.floor((line_thickness - 1)/2)

                # LEFT BORDER

                for i in range(top, bot):
                    border = max(left - it, 0)
                    for j in range(border, left + it + 1):
                        bound_map[i, j] = 1

                # right border

                for i in range(top, bot):
                    border = min(right + it, OUTPUT_SIZE - 1)
                    for j in range(right - it, border + 1):
                        bound_map[i, j] = 1

                # top border

                for j in range(left, right):
                    border = max(top - it, 0)
                    for i in range(border, top + it + 1):
                        bound_map[i, j] = 1

                # bot_border

                for j in range(left, right):
                    border = min(bot + it, OUTPUT_SIZE - 1)
                    for i in range(bot - it, border + 1):
                        bound_map[i, j] = 1
            #stop = True
    #print(bound_map.max())
    for row in range(INPUT_SIZE):
        for pil in range(INPUT_SIZE):
            if bound_map[row, pil] == 1:
                #print(row, pil)
                dt[row, pil, 0] = 255
                dt[row, pil, 1] = 0
                dt[row, pil, 2] = 0

    IMG = Image.fromarray(COEF * bound_map.astype('uint8'), 'L')
    IMG.show()
    IMG = Image.fromarray(dt.astype('uint8'), 'RGB')
    IMG.show()

boundary_maker()