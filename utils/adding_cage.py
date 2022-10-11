import math
import ntpath
import os
import os.path
import random

import networkx as nx
from shapely.geometry import Polygon
import cv2
import numpy as np
import argparse
# import globalHandler


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_radius', type=str, help='min radius for blurring image, between 0-9')
    parser.add_argument('--min_rotation', type=str, help='min angle for rotating image, between -180,+180')
    parser.add_argument('--max_radius', type=str, help='max radius for blurring image, between 0-9')
    parser.add_argument('--max_rotation', type=str, help='min angle for rotating image, between -180,+180')
    parser.add_argument('--min_cageScaling', type=str, help='min multiplier for scaling cage, between 2,4')
    parser.add_argument('--max_cageScaling', type=str, help='max multiplier for scaling cage, between 2,4')
    parser.add_argument('--cage', type=str, help='cage name')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    return background


def ModifiedWay(rotateImage, angle):
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]
    centreY, centreX = imgHeight // 2, imgWidth // 2
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)
    cosRotationMatrix = np.abs(rotationMatrix[0][0])
    sinRotationMatrix = np.abs(rotationMatrix[0][1])
    newImageHeight = int((imgHeight * sinRotationMatrix) +
                         (imgWidth * cosRotationMatrix))
    newImageWidth = int((imgHeight * cosRotationMatrix) +
                        (imgWidth * sinRotationMatrix))

    rotationMatrix[0][2] += (newImageWidth / 2) - centreX
    rotationMatrix[1][2] += (newImageHeight / 2) - centreY

    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight))

    return rotatingimage


def length(x, y):
    return math.sqrt(x ** 2 + y ** 2)


def merge_box(image, path):
    opt = parse_opt(known=True)
    List_cage = opt.cage.split("-")
    # List_cage = globalHandler.hyperParams[6].split("-")
    Cage_Address = "/workspace16/Shayan/DataSet/cage-photo/" + str(random.choice(List_cage)) + ".png"
    logo = cv2.imread(Cage_Address, cv2.IMREAD_UNCHANGED)

    try:
        size = image.shape
        name = ntpath.basename(path).split('.')[0]
        change_factor = random.uniform(float(opt.min_cageScaling), float(opt.max_cageScaling))
        # change_factor = random.uniform(float(globalHandler.hyperParams[4]), float(globalHandler.hyperParams[5]))
        
        with open(os.path.dirname(os.path.abspath(path)).replace("images", "labels") + "/" + name + ".txt") as f_label:
            x_center_arr = []
            y_center_arr = []
            width_arr = []
            height_arr = []
            list_edge = []
            coords = []

            for line in f_label:
                hold = line.split()
                x_center_arr.append(float(hold[1]) * float(size[1]))
                y_center_arr.append(float(hold[2]) * float(size[0]))
                width_arr.append((float(hold[3]) * change_factor) * float(size[1]))
                height_arr.append((float(hold[4]) * change_factor) * float(size[0]))

            for x in range(len(x_center_arr)):
                x1 = int(x_center_arr[x] - width_arr[x] / 2)
                x2 = int(x_center_arr[x] + width_arr[x] / 2)
                y1 = int(y_center_arr[x] - height_arr[x] / 2)
                y2 = int(y_center_arr[x] + height_arr[x] / 2)

                left_down = (x1, y1)
                left_up = (x1, y2)
                right_down = (x2, y1)
                right_up = (x2, y2)

                coord = [left_up, right_up, right_down, left_down, left_up]
                coords.append(coord)

            for item1 in range(len(coords)):
                p1 = Polygon(coords[item1])
                for item2 in range(len(coords)):
                    p2 = Polygon(coords[item2])
                    if p1.intersects(p2) & (item1 != item2):
                        list_edge.append([item1, item2])

            G = nx.from_edgelist(list_edge)
            G.add_nodes_from(range(len(coords)))
            connected = list(nx.connected_components(G))

            list_box = []
            for item1 in connected:
                p_hold = Polygon([])
                for item2 in item1:
                    p_hold = p_hold.union(Polygon(coords[item2]))
                box = p_hold.convex_hull.bounds
                list_box.append(box)
            image_copy = image.copy()

            hold_rand = 2 * (random.randint(int(opt.min_radius), int(opt.max_radius)) // 2) + 1
            # hold_rand = 2 * (random.randint(int(globalHandler.hyperParams[0]), int(globalHandler.hyperParams[1])) // 2) + 1
            
            for item in list_box:
                width = abs(item[2] - item[0])
                height = abs(item[3] - item[1])
                x_center = int(item[0])
                y_center = int(item[1])

                logo_image = cv2.GaussianBlur(ModifiedWay(logo, random.randint(int(opt.min_rotation), int(opt.max_rotation))),
                                              (hold_rand, hold_rand), 0)
                # logo_image = cv2.GaussianBlur(ModifiedWay(logo, random.randint(int(globalHandler.hyperParams[2]), int(globalHandler.hyperParams[3]))),
                #                               (hold_rand, hold_rand), 0)
                
                logo_image = cv2.resize(logo_image, (int(width), int(height)))

                image_copy = add_transparent_image(image_copy, logo_image, x_center, y_center)
            return cv2.GaussianBlur(image_copy, (hold_rand, hold_rand), 0)
    except Exception as e:
        print(e)


