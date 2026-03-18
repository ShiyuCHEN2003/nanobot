#!/usr/bin/env python3
"""生成 ChArUco 标定板并保存为 PNG，打印出来用于标定。"""

import cv2
import numpy as np

COLS = 7       # 棋盘列数（角点数 = COLS-1）
ROWS = 5       # 棋盘行数
SQUARE_M = 0.030   # 棋格边长 30mm
MARKER_M = 0.022   # ArUco marker 边长 22mm
DICT_ID  = cv2.aruco.DICT_4X4_50
OUT      = "charuco_board.png"
DPI      = 300
# A4 @ 300dpi = 2480 × 3508 px；留白后有效区域约 2100×1485
IMG_W, IMG_H = 2100, 1485

dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_M, MARKER_M, dictionary)
img = board.generateImage((IMG_W, IMG_H), marginSize=20, borderBits=1)
cv2.imwrite(OUT, img)
print(f"标定板已保存到 {OUT}，请用 A4 纸打印（实际比例 1:1，不要缩放）")
print(f"棋格参数: {COLS}×{ROWS}，方格 {SQUARE_M*1000:.0f}mm，marker {MARKER_M*1000:.0f}mm")
