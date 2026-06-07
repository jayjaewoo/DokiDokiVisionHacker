# -*- coding: utf-8 -*-
"""
Vision Face Hacker: V-Line Face Simulator (Pure Python 통합판)
====================================================================
PyQt5 단일 프레임워크로 비주얼 노벨 스토리 엔진과
OpenCV/MediaPipe 컴퓨터 비전 미니게임을 완전 통합한 프로그램입니다.
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions as MpBaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker, FaceLandmarkerOptions
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as RunningMode

# =========================================================================
# 리소스 경로 설정
# - .py 실행: main.py가 있는 폴더 기준
# - PyInstaller exe 실행: exe가 있는 폴더 기준
# - PyInstaller onefile 내부 번들: sys._MEIPASS도 보조 탐색
# =========================================================================
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_PATH = get_base_path()
BUNDLE_PATH = getattr(sys, "_MEIPASS", BASE_PATH)


def resource_path(relative_path):
    """외부 폴더와 PyInstaller 번들 내부를 모두 탐색하는 안전한 경로 함수."""
    candidates = [
        os.path.join(BASE_PATH, relative_path),
        os.path.join(BUNDLE_PATH, relative_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# MediaPipe 모델 파일 경로
MODEL_PATH = resource_path("face_landmarker.task")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QVBoxLayout, QHBoxLayout,
    QProgressBar, QFrame, QStackedWidget, QSizePolicy, QGraphicsOpacityEffect
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QLinearGradient, QPalette, QBrush, QPolygon
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QPoint
)

# =========================================================================
# 이미지 에셋 경로
# =========================================================================
IMAGE_DIR = resource_path("images")


def get_img_path(filename):
    """images 폴더에서 이미지 파일의 절대 경로를 반환. 없으면 None."""
    if filename is None:
        return None
    path = os.path.join(IMAGE_DIR, filename)
    return path if os.path.exists(path) else None


# =========================================================================
# [MODIFY_DIALOGUE] 스토리 시나리오 데이터
# =========================================================================
# 각 노드 키:
#   speaker  : 화자 이름
#   text     : 대사 텍스트
#   bg       : 배경 이미지 파일명 (None이면 이전 배경 유지)
#   sprite   : 캐릭터 스프라이트 파일명 (None이면 숨김)
#   next     : 다음 노드 인덱스
#   action   : 특수 액션 ("MINIGAME", "CHOICE", "END")
# =========================================================================

STORY = [
    # 0  (원본 0)
    {'speaker': '시스템',
     'text': '코로나19 재확산으로 인한 원내 마스크 착용 의무화 기간.\n강의실 안의 모든 사람들은 철저히 마스크를 쓰고 다닌다.',
     'bg': 'classroom_bg.png',
     'sprite': None,
     'next': 1},
    # 1  (원본 1)
    {'speaker': '나',
     'text': "그리고 내 앞자리에 앉은 그녀, 임희지.\n그녀는 학교에서 알아주는 극도의 외모 지상주의자, 이른바 '얼빠'이다.",
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 2},
    # 2  (원본 2 / part 1)
    {'speaker': '희지',
     'text': '저기... 너 이번 주말에 약속 있어?\n매일 마스크 쓴 모습만 보니까 답답해서.',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 3},
    # 3  (원본 2 / part 2)
    {'speaker': '희지',
     'text': '주말에 카페에서 만날래?',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 4},
    # 4  (원본 3)
    {'speaker': '희지',
     'text': '만나기 전에, 네 마스크 벗은 생얼 사진 좀 먼저 보내줘!\n눈 크고 얼굴 대칭이고 V라인인지 미리 스캔해보고 싶거든!',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_excited.png',
     'next': 5},
    # 5  (원본 4)
    {'speaker': '나',
     'text': '(망했다. 내 생얼은 눈 작고 비대칭이고 넙대대한데...)\n(방법은 단 하나, 비전 픽셀 유동화 알고리즘으로 내 얼굴을 성형해서 보내자!)',
     'bg': 'classroom_bg.png',
     'sprite': None,
     'action': 'MINIGAME'},
    # 6  (원본 5)
    {'speaker': '나',
     'text': '좋아, 완벽하다! 이 정도 대칭과 비율의 아바타라면 의심하지 않겠지.\n보정된 해킹 사진을 희지에게 전송했다.',
     'bg': 'classroom_bg.png',
     'sprite': None,
     'next': 7},
    # 7  (원본 6 / part 1)
    {'speaker': '희지',
     'text': '와! 대박! 너 진짜 이목구비 깡패구나?\n얼굴 라인이 컴퓨터처럼 완벽해!',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_excited.png',
     'next': 8},
    # 8  (원본 6 / part 2)
    {'speaker': '희지',
     'text': '이번 주말 데이트, 정말 기대된다!',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_excited.png',
     'next': 9},
    # 9  (원본 7)
    {'speaker': '시스템',
     'text': '[ 주말. 분위기 좋은 카페 ]',
     'bg': 'cafe_bg.png',
     'sprite': None,
     'next': 10},
    # 10  (원본 8)
    {'speaker': '나',
     'text': '우리는 마스크를 쓴 채 마주 앉았다.\n희지는 상기된 얼굴로 나를 본다.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_excited.png',
     'next': 11},
    # 11  (원본 9)
    {'speaker': '희지',
     'text': '사진 속 그 완벽한 얼굴을 드디어 보네.\n이제 마스크 벗고 얼굴 보여줘!',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_excited.png',
     'action': 'CHOICE'},
    # 12  (원본 10)
    {'speaker': '나',
     'text': '나는 꿀꺽 침을 삼키며 마스크를 완전히 벗었다.',
     'bg': 'cafe_bg.png',
     'sprite': None,
     'next': 13},
    # 13  (원본 11)
    {'speaker': '희지',
     'text': '……아.\n이게 네 얼굴이구나.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 14},
    # 14  (원본 12 / part 1)
    {'speaker': '희지',
     'text': '이상해.\n분명 지금 처음 본 건데,',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 15},
    # 15  (원본 12 / part 2)
    {'speaker': '희지',
     'text': '나는 방금 뭔가를 잃어버린 기분이 들어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 16},
    # 16  (원본 13 / part 1)
    {'speaker': '희지',
     'text': '아니.\n정확히는,',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 17},
    # 17  (원본 13 / part 2)
    {'speaker': '희지',
     'text': '네가 죽은 것 같아.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 18},
    # 18  (원본 14 / part 1)
    {'speaker': '희지',
     'text': 'Gustave Moreau, Orphée.\n그 그림에서 오르페우스는 이미 죽었잖아.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 19},
    # 19  (원본 14 / part 2)
    {'speaker': '희지',
     'text': '몸은 사라지고, 머리만 남아 있어.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 20},
    # 20  (원본 15 / part 1)
    {'speaker': '희지',
     'text': '그런데도 이상하게 아름다워.\n아니, 죽었기 때문에 아름다워.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 21},
    # 21  (원본 15 / part 2)
    {'speaker': '희지',
     'text': '더 이상 말하지 않고,\n더 이상 실망시키지 않고,',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 22},
    # 22  (원본 15 / part 3)
    {'speaker': '희지',
     'text': '더 이상 인간으로 돌아오지 않으니까.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 23},
    # 23  (원본 16 / part 1)
    {'speaker': '희지',
     'text': '나는 네가 그런 식으로 남아 있길 바랐나 봐.\n살아 있는 남자가 아니라.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 24},
    # 24  (원본 16 / part 2)
    {'speaker': '희지',
     'text': '내 머릿속에서만 노래하는 잘린 머리.\n얼굴만 있고 몸은 없는 사람.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 25},
    # 25  (원본 16 / part 3)
    {'speaker': '희지',
     'text': '실재 없이 아름다운 남자.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 26},
    # 26  (원본 17 / part 1)
    {'speaker': '희지',
     'text': '……웃기지?\n나는 네 얼굴을 보고 싶다고 생각했어.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 27},
    # 27  (원본 17 / part 2)
    {'speaker': '희지',
     'text': "그런데 정말 보고 싶었던 건 얼굴이 아니었어.\n'볼 수 없음'이었어.",
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 28},
    # 28  (원본 17 / part 3)
    {'speaker': '희지',
     'text': '금기.\n닫힌 문.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 29},
    # 29  (원본 17 / part 4)
    {'speaker': '희지',
     'text': '마스크 아래에 있을지도 모르는,\n아직 더럽혀지지 않은 가능성.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 30},
    # 30  (원본 18 / part 1)
    {'speaker': '희지',
     'text': '조르주 바타유는 말했지.\n금기는 금지하기 위해 있는 게 아니라,',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 31},
    # 31  (원본 18 / part 2)
    {'speaker': '희지',
     'text': '위반되기 위해 존재한다고.\n그런데 위반은 축제가 아니야.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 32},
    # 32  (원본 18 / part 3)
    {'speaker': '희지',
     'text': '제의야.\n무언가를 바쳐야만 완성되는 제의.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 33},
    # 33  (원본 18 / part 4)
    {'speaker': '희지',
     'text': '그리고 지금 제물은…\n내가 사랑하던 네 얼굴이었어.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 34},
    # 34  (원본 19 / part 1)
    {'speaker': '희지',
     'text': '아니.\n네 얼굴이라는 환상이었어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 35},
    # 35  (원본 19 / part 2)
    {'speaker': '희지',
     'text': '내가 사랑한 건 네가 아니었어.\n네 안에 있다고 믿었던 결핍.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 36},
    # 36  (원본 19 / part 3)
    {'speaker': '희지',
     'text': '아직 열리지 않은 상자.\n내가 감히 넘볼 수 없던 성소.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 37},
    # 37  (원본 19 / part 4)
    {'speaker': '희지',
     'text': '그 어둠이었어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 38},
    # 38  (원본 20 / part 1)
    {'speaker': '희지',
     'text': '그런데 네가 그걸 직접 열어버렸네.\n너무 쉽게.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 39},
    # 39  (원본 20 / part 2)
    {'speaker': '희지',
     'text': '너무 선하게.\n너무 인간답게.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 40},
    # 40  (원본 20 / part 3)
    {'speaker': '희지',
     'text': '그래서 끝난 거야.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 41},
    # 41  (원본 21 / part 1)
    {'speaker': '희지',
     'text': '진실이 나빠서가 아니야.\n네가 못생겨서도 아니야.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 42},
    # 42  (원본 21 / part 2)
    {'speaker': '희지',
     'text': '그런 말로는 설명이 안 돼.\n욕망은 원래 대상을 원하지 않아.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 43},
    # 43  (원본 21 / part 3)
    {'speaker': '희지',
     'text': '욕망은 대상을 둘러싼 금지를 원해.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 44},
    # 44  (원본 22 / part 1)
    {'speaker': '희지',
     'text': '손댈 수 없다는 거리.\n보면 안 된다는 떨림.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 45},
    # 45  (원본 22 / part 2)
    {'speaker': '희지',
     'text': '닿는 순간 끝난다는 공포.\n나는 그 공포를 사랑했어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 46},
    # 46  (원본 22 / part 3)
    {'speaker': '희지',
     'text': '너를 사랑한 게 아니라.\n네가 끝내 보여주지 않을 거라고 믿었던 것을 사랑했어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 47},
    # 47  (원본 23 / part 1)
    {'speaker': '희지',
     'text': '그래서 지금 네 얼굴은 얼굴이 아니야.\n참수야.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 48},
    # 48  (원본 23 / part 2)
    {'speaker': '희지',
     'text': '내 환상이 자기 목을 잃은 장면이야.\nGustave Moreau, Orphée.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 49},
    # 49  (원본 24 / part 1)
    {'speaker': '희지',
     'text': '잘린 머리는 아직 아름답지만,\n그 머리가 다시 몸을 얻는 순간',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 50},
    # 50  (원본 24 / part 2)
    {'speaker': '희지',
     'text': '노래는 끝나.\n네가 내 앞에서 살아 있는 사람이 된 순간,',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 51},
    # 51  (원본 24 / part 3)
    {'speaker': '희지',
     'text': '나는 더 이상 너를 들을 수 없게 됐어.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 52},
    # 52  (원본 25 / part 1)
    {'speaker': '희지',
     'text': '미안해.\n정말 미안해.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 53},
    # 53  (원본 25 / part 2)
    {'speaker': '희지',
     'text': '하지만 나는 네 진실보다,\n네가 숨기고 있던 허구를 더 사랑했어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 54},
    # 54  (원본 25 / part 3)
    {'speaker': '희지',
     'text': '그러니까 제발…\n내 기억 속에서는 다시 마스크를 써줘.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 55},
    # 55  (원본 26 / part 1)
    {'speaker': '희지',
     'text': '내 안에서만큼은\n계속 죽어 있어줘.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 56},
    # 56  (원본 26 / part 2)
    {'speaker': '희지',
     'text': '아름답게.\n말없이.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 57},
    # 57  (원본 26 / part 3)
    {'speaker': '희지',
     'text': '끝내 벗겨지지 않는 얼굴로.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 58},
    # 58  (원본 27 / part 1)
    {'speaker': '나',
     'text': '희지는 뒤돌아 울며 가버렸다.\n나의 진짜 얼굴은 절대로 사랑받지 못했다.',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'next': 59},
    # 59  (원본 27 / part 2)
    {'speaker': '나',
     'text': '【ENDING A: 바타유와 모로 — 머리 없는 인간】',
     'bg': 'cafe_bg.png',
     'sprite': 'orphee.png',
     'action': 'END'},
    # 60  (원본 28)
    {'speaker': '나',
     'text': '희지야, 미안하지만 나는 마스크를 벗지 않을게.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 61},
    # 61  (원본 29 / part 1)
    {'speaker': '희지',
     'text': '……잠깐.\n벗기지 마.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 62},
    # 62  (원본 29 / part 2)
    {'speaker': '희지',
     'text': '이상하게 이제 알 것 같아.\n내가 정말 보고 싶었던 게 뭔지.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 63},
    # 63  (원본 30 / part 1)
    {'speaker': '희지',
     'text': 'René Magritte, Les Amants.\n얼굴은 가려져 있는데,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 64},
    # 64  (원본 30 / part 2)
    {'speaker': '희지',
     'text': '우리는 그걸 연인이라고 부르잖아.\n이상하지.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 65},
    # 65  (원본 30 / part 3)
    {'speaker': '희지',
     'text': '아무것도 드러나지 않았는데,\n오히려 너무 정확해.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 66},
    # 66  (원본 31 / part 1)
    {'speaker': '희지',
     'text': '사랑이라는 건 어쩌면 얼굴을 확인하는 일이 아니라,\n얼굴이 있다고 믿는 방식인지도 몰라.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 67},
    # 67  (원본 31 / part 2)
    {'speaker': '희지',
     'text': '네가 나를 속였다는 건 알아.\n그런데 내가 사랑한 것도 처음부터 진실은 아니었어.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 68},
    # 68  (원본 32 / part 1)
    {'speaker': '희지',
     'text': '나는 네 얼굴을 사랑한 게 아니야.\n네 얼굴이라는 이미지.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 69},
    # 69  (원본 32 / part 2)
    {'speaker': '희지',
     'text': '네가 보여준 표면.\n내가 거기에 덧씌운 의미.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 70},
    # 70  (원본 32 / part 3)
    {'speaker': '희지',
     'text': '그 모든 허구의 질감을 사랑했어.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 71},
    # 71  (원본 33 / part 1)
    {'speaker': '희지',
     'text': '보드리야르는 원본이 사라진 시대를 말했지.\n기호가 대상을 가리키는 게 아니라,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 72},
    # 72  (원본 33 / part 2)
    {'speaker': '희지',
     'text': '기호가 대상을 대신하는 세계.\n아니.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 73},
    # 73  (원본 33 / part 3)
    {'speaker': '희지',
     'text': '대신하는 정도가 아니야.\n원본보다 더 원본 같은 것.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 74},
    # 74  (원본 33 / part 4)
    {'speaker': '희지',
     'text': '현실보다 더 끈질긴 현실.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 75},
    # 75  (원본 34 / part 1)
    {'speaker': '희지',
     'text': '네 마스크가 그랬어.\n처음엔 가짜였겠지.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 76},
    # 76  (원본 34 / part 2)
    {'speaker': '희지',
     'text': '그런데 내가 매일 바라보고,\n매일 믿고,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 77},
    # 77  (원본 34 / part 3)
    {'speaker': '희지',
     'text': '매일 사랑하는 동안\n그 가짜는 더 이상 가짜가 아니게 됐어.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 78},
    # 78  (원본 34 / part 4)
    {'speaker': '희지',
     'text': '네 진짜 얼굴보다,\n이 마스크가 먼저 내 현실이 됐어.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 79},
    # 79  (원본 35 / part 1)
    {'speaker': '희지',
     'text': '그러니까 지금 벗기는 게 오히려 거짓말일지도 몰라.\n원본을 보여주는 행위가 진실인 척하는 가장 낡은 연극일지도 몰라.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 80},
    # 80  (원본 35 / part 2)
    {'speaker': '희지',
     'text': '나는 이제 알고 싶지 않아.\n네가 누구인지.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 81},
    # 81  (원본 35 / part 3)
    {'speaker': '희지',
     'text': '네가 어떻게 생겼는지.\n네 피부 아래에 어떤 실재가 있는지.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 82},
    # 82  (원본 36 / part 1)
    {'speaker': '희지',
     'text': '그런 건 너무 가난해.\n너무 단순해.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 83},
    # 83  (원본 36 / part 2)
    {'speaker': '희지',
     'text': '너무 생물학적이야.\n내가 사랑한 너는 그런 얼굴 하나로 환원될 수 없어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 84},
    # 84  (원본 37 / part 1)
    {'speaker': '희지',
     'text': '너는 이미 이미지가 됐어.\n기호가 됐고,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 85},
    # 85  (원본 37 / part 2)
    {'speaker': '희지',
     'text': '약속이 됐고,\n반복되는 화면이 됐고,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 86},
    # 86  (원본 37 / part 3)
    {'speaker': '희지',
     'text': '내가 사랑을 인식하는 방식 자체가 됐어.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 87},
    # 87  (원본 38 / part 1)
    {'speaker': '희지',
     'text': '저 둘은 서로를 보지 못해.\n그런데도 키스하지.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 88},
    # 88  (원본 38 / part 2)
    {'speaker': '희지',
     'text': '어쩌면 사랑은 서로의 얼굴을 확인하는 순간 시작되는 게 아니라,\n끝내 확인하지 않기로 합의하는 순간 지속되는 건지도 몰라.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 89},
    # 89  (원본 39 / part 1)
    {'speaker': '희지',
     'text': '좋아.\n벗지 마.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 90},
    # 90  (원본 39 / part 2)
    {'speaker': '희지',
     'text': '이제는 내가 원하지 않아.\n너의 원본은 필요 없어.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 91},
    # 91  (원본 39 / part 3)
    {'speaker': '희지',
     'text': '원본은 언제나 늦게 도착하니까.',
     'bg': 'cafe_bg.png',
     'sprite': 'heeji_normal.png',
     'next': 92},
    # 92  (원본 40 / part 1)
    {'speaker': '희지',
     'text': '이미지는 이미 나를 사랑하게 만들었고,\n실재는 그 뒤에 와서 변명할 뿐이야.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 93},
    # 93  (원본 40 / part 2)
    {'speaker': '희지',
     'text': '그러니까 계속 그 얼굴로 있어.\n내가 사랑한 얼굴로.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 94},
    # 94  (원본 40 / part 3)
    {'speaker': '희지',
     'text': '네 것이 아니지만,\n이제는 우리 것이 된 얼굴로.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 95},
    # 95  (원본 41 / part 1)
    {'speaker': '희지',
     'text': '이건 거짓말이 아니야.\n거짓말은 아직 진실을 두려워하지.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 96},
    # 96  (원본 41 / part 2)
    {'speaker': '희지',
     'text': '하지만 우리는 진실을 대체했어.\n네 마스크는 숨기는 물건이 아니야.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 97},
    # 97  (원본 41 / part 3)
    {'speaker': '희지',
     'text': '네 얼굴이야.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 98},
    # 98  (원본 42 / part 1)
    {'speaker': '희지',
     'text': '그리고 내가 사랑하는 건\n그 아래에 있는 사람이 아니라,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 99},
    # 99  (원본 42 / part 2)
    {'speaker': '희지',
     'text': '그 위에 남아 있는 너야.\n평생 벗지 마.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 100},
    # 100  (원본 42 / part 3)
    {'speaker': '희지',
     'text': '나도 평생 묻지 않을게.\n우리가 같은 허구를 믿는 동안,',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 101},
    # 101  (원본 42 / part 4)
    {'speaker': '희지',
     'text': '그 허구는 현실보다 오래 갈 테니까.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 102},
    # 102  (원본 43 / part 1)
    {'speaker': '희지',
     'text': '사랑해.\n네가 아니라.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 103},
    # 103  (원본 43 / part 2)
    {'speaker': '희지',
     'text': '네가 된 이미지가.',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 104},
    # 104  (원본 44 / part 1)
    {'speaker': '나',
     'text': '우리는 끝내 마스크를 벗지 않은 채,\n천을 감싸고 키스하는 마그리트의 그림처럼',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'next': 105},
    # 105  (원본 44 / part 2)
    {'speaker': '나',
     'text': '기묘한 사랑을 완성했다.\n【ENDING B: 보드리야르와 마그리트 — 시뮬라크르의 지속】',
     'bg': 'cafe_bg.png',
     'sprite': 'les_amants.png',
     'action': 'END'},
    # 106  (원본 45)
    {'speaker': '나',
     'text': '해킹에 실패했다. 얼굴 왜곡이 심해 미니게임을 끝내지 못했다.',
     'bg': 'classroom_bg.png',
     'sprite': None,
     'next': 107},
    # 107  (원본 46 / part 1)
    {'speaker': '희지',
     'text': '사진 왜 안 보내줘? 얼굴에 자신 없나 보네.\n실망이야. 바이.',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_sad.png',
     'next': 108},
    # 108  (원본 46 / part 2)
    {'speaker': '희지',
     'text': '【BAD ENDING: 외모 검열 탈락】',
     'bg': 'classroom_bg.png',
     'sprite': 'heeji_sad.png',
     'action': 'END'},
]

# Y 엔딩 시작 노드 / N 엔딩 시작 노드
Y_ENDING_START = 12
N_ENDING_START = 60
MINIGAME_SUCCESS_NODE = 6
MINIGAME_FAIL_NODE = 106


# =========================================================================
# 픽셀 유동화 알고리즘 (Pixel Liquify via cv2.remap)
# =========================================================================
def warp_image(img, start_x, start_y, end_x, end_y, radius=45, strength=0.45):
    """마우스 드래그 방향으로 브러시 반경 내 픽셀을 유동화합니다."""
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    dx = end_x - start_x
    dy = end_y - start_y
    if np.sqrt(dx ** 2 + dy ** 2) < 1:
        return img.copy()

    dist_map = np.sqrt((map_x - start_x) ** 2 + (map_y - start_y) ** 2)
    mask = dist_map < radius
    weight = (1.0 - (dist_map / radius) ** 2) ** 2 * strength
    weight[~mask] = 0

    map_x_new = np.clip(map_x - weight * dx, 0, w - 1)
    map_y_new = np.clip(map_y - weight * dy, 0, h - 1)

    return cv2.remap(img, map_x_new, map_y_new, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def pt_dist(p1, p2):
    """두 점 사이의 유클리드 거리."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# =========================================================================
# Style Constants
# =========================================================================
DARK_BG = "#0d0d14"
PANEL_BG = "rgba(15, 15, 25, 0.92)"
ACCENT_GREEN = "#00ff66"
ACCENT_CYAN = "#00ffff"
ACCENT_PINK = "#ff66cc"
ACCENT_YELLOW = "#ffd700"
FONT_MAIN = "Malgun Gothic"
FONT_MONO = "Consolas"
CANVAS_W = 860
CANVAS_H = 645
SELECTION_SCALE_STEP = 1.18


# =========================================================================
# Story Widget (Visual Novel Engine)
# =========================================================================
class StoryWidget(QWidget):
    """비주얼 노벨 스토리 엔진 위젯."""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.current_idx = 0
        self.current_bg = None  # 현재 배경 파일명 기억
        self.typing_timer = QTimer(self)
        self.typing_timer.setInterval(30)
        self.typing_timer.timeout.connect(self._type_next_char)
        self.full_text = ""
        self.displayed_text = ""
        self.char_index = 0
        self.is_typing = False
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(f"background-color: {DARK_BG};")
        self.setFocusPolicy(Qt.StrongFocus)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 뷰포트 영역 (배경 + 캐릭터) ──
        self.viewport = QLabel(self)
        self.viewport.setAlignment(Qt.AlignCenter)
        self.viewport.setStyleSheet("background-color: #111119;")
        self.viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.viewport, stretch=1)

        # ── 대사창 프레임 ──
        self.textbox_frame = QFrame(self)
        self.textbox_frame.setFixedHeight(250)
        self.textbox_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL_BG};
                border-top: 3px solid {ACCENT_GREEN};
            }}
        """)
        vbox = QVBoxLayout(self.textbox_frame)
        vbox.setContentsMargins(28, 14, 28, 14)
        vbox.setSpacing(6)

        # 화자 이름
        self.speaker_label = QLabel("")
        self.speaker_label.setFont(QFont(FONT_MONO, 15, QFont.Bold))
        self.speaker_label.setStyleSheet(
            f"color: {ACCENT_PINK}; border: none; background: transparent;")
        vbox.addWidget(self.speaker_label)

        # 대사 텍스트
        self.text_label = QLabel("")
        self.text_label.setFont(QFont(FONT_MAIN, 12))
        self.text_label.setStyleSheet(
            "color: #e0e0e0; border: none; background: transparent;")
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        vbox.addWidget(self.text_label, stretch=1)

        # ── 선택지 버튼 ──
        self.choice_layout = QHBoxLayout()
        self.choice_layout.setSpacing(16)
        self.btn_y = QPushButton("용기를 내어 마스크를 벗고\n진짜 내 얼굴을 보여준다. (Y)")
        self.btn_n = QPushButton("마스크를 벗지 않고\n가상과 약속의 관계를 유지한다. (N)")
        # 선택 후 Space/Enter가 버튼을 다시 누르는 현상을 막습니다.
        self.btn_y.setFocusPolicy(Qt.NoFocus)
        self.btn_n.setFocusPolicy(Qt.NoFocus)
        self.btn_y.setMinimumHeight(64)
        self.btn_n.setMinimumHeight(64)
        btn_style = f"""
            QPushButton {{
                background-color: #1a1a28;
                color: {ACCENT_CYAN};
                border: 2px solid {ACCENT_CYAN};
                border-radius: 6px;
                padding: 12px 8px;
                font-weight: bold;
                font-family: '{FONT_MAIN}';
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_CYAN};
                color: #000;
            }}
        """
        self.btn_y.setStyleSheet(btn_style)
        self.btn_n.setStyleSheet(btn_style)
        self.btn_y.clicked.connect(lambda: self.make_choice("Y"))
        self.btn_n.clicked.connect(lambda: self.make_choice("N"))
        self.choice_layout.addWidget(self.btn_y)
        self.choice_layout.addWidget(self.btn_n)
        self.btn_y.hide()
        self.btn_n.hide()
        vbox.addLayout(self.choice_layout)

        # 클릭 안내
        self.hint_label = QLabel("▶ Click / Space to continue")
        self.hint_label.setFont(QFont(FONT_MONO, 9))
        self.hint_label.setStyleSheet("color: #555; border: none; background: transparent;")
        self.hint_label.setAlignment(Qt.AlignRight)
        vbox.addWidget(self.hint_label)

        main_layout.addWidget(self.textbox_frame)

    # ── 타이핑 애니메이션 ──
    def _start_typing(self, text):
        self.full_text = text
        self.displayed_text = ""
        self.char_index = 0
        self.is_typing = True
        self.text_label.setText("")
        self.hint_label.setText("")
        self.typing_timer.start()

    def _type_next_char(self):
        if self.char_index < len(self.full_text):
            self.displayed_text += self.full_text[self.char_index]
            self.text_label.setText(self.displayed_text)
            self.char_index += 1
        else:
            self.typing_timer.stop()
            self.is_typing = False
            node = STORY[self.current_idx]
            action = node.get("action")
            if action != "CHOICE" and action != "END":
                self.hint_label.setText("▶ Click / Space to continue")

    def _skip_typing(self):
        """타이핑 스킵 → 전문 즉시 표시."""
        self.typing_timer.stop()
        self.text_label.setText(self.full_text)
        self.displayed_text = self.full_text
        self.is_typing = False
        node = STORY[self.current_idx]
        action = node.get("action")
        if action != "CHOICE" and action != "END":
            self.hint_label.setText("▶ Click / Space to continue")

    # ── 노드 로드 ──
    def load_node(self, idx):
        self.current_idx = idx
        node = STORY[idx]

        # 화자명 + 색상
        speaker = node.get("speaker", "")
        self.speaker_label.setText(speaker)
        if speaker == "희지":
            self.speaker_label.setStyleSheet(
                f"color: {ACCENT_PINK}; border: none; background: transparent;")
        elif speaker == "나":
            self.speaker_label.setStyleSheet(
                f"color: {ACCENT_CYAN}; border: none; background: transparent;")
        else:
            self.speaker_label.setStyleSheet(
                f"color: {ACCENT_GREEN}; border: none; background: transparent;")

        # 배경 처리 (None이면 이전 배경 유지)
        bg_file = node.get("bg")
        if bg_file is not None:
            self.current_bg = bg_file

        # 스프라이트/배경 렌더링
        self._render_viewport(node.get("sprite"))

        # 대사 타이핑 시작
        self._start_typing(node.get("text", ""))

        # 선택지 표시/숨김
        action = node.get("action")
        if action == "CHOICE":
            self.btn_y.setEnabled(True)
            self.btn_n.setEnabled(True)
            self.btn_y.show()
            self.btn_n.show()
            self.hint_label.hide()
        else:
            self.btn_y.setEnabled(False)
            self.btn_n.setEnabled(False)
            self.btn_y.hide()
            self.btn_n.hide()
            self.hint_label.show()

    def _render_viewport(self, sprite_filename):
        """배경 위에 스프라이트를 합성하여 뷰포트에 표시."""
        vp_w = self.viewport.width() if self.viewport.width() > 100 else 960
        vp_h = self.viewport.height() if self.viewport.height() > 100 else 400

        # 배경 로드
        canvas = QPixmap(vp_w, vp_h)
        canvas.fill(QColor("#111119"))

        if self.current_bg:
            bg_path = get_img_path(self.current_bg)
            if bg_path:
                bg_pix = QPixmap(bg_path).scaled(
                    vp_w, vp_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                painter = QPainter(canvas)
                # 중앙 정렬 크롭
                x_off = (bg_pix.width() - vp_w) // 2
                y_off = (bg_pix.height() - vp_h) // 2
                painter.drawPixmap(0, 0, bg_pix, x_off, y_off, vp_w, vp_h)
                painter.end()

        # 스프라이트 오버레이
        if sprite_filename:
            sp_path = get_img_path(sprite_filename)
            if sp_path:
                sp_pix = QPixmap(sp_path).scaled(
                    vp_h, vp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter = QPainter(canvas)
                x = (vp_w - sp_pix.width()) // 2
                y = vp_h - sp_pix.height()
                painter.drawPixmap(x, y, sp_pix)
                painter.end()
            else:
                # 이미지 없으면 텍스트 표시
                painter = QPainter(canvas)
                painter.setPen(QColor("#ff3333"))
                painter.setFont(QFont(FONT_MONO, 11))
                painter.drawText(canvas.rect(), Qt.AlignCenter,
                                 f"[Image Missing: {sprite_filename}]")
                painter.end()

        self.viewport.setPixmap(canvas)

    # ── 진행 ──
    def advance(self):
        if self.is_typing:
            self._skip_typing()
            return

        node = STORY[self.current_idx]
        action = node.get("action")

        if action == "CHOICE":
            return  # 선택지 대기
        elif action == "MINIGAME":
            self.parent_app.start_minigame()
        elif action == "END":
            # 엔딩 도달 → 타이틀로 돌아가기
            self.parent_app.show_title()
        else:
            next_idx = node.get("next")
            if next_idx is not None:
                self.load_node(next_idx)

    def make_choice(self, choice):
        # 현재 노드가 선택지일 때만 엔딩 분기로 들어갑니다.
        # 선택 버튼이 포커스를 가진 채 Space/Enter로 다시 눌려 같은 노드가 반복되는 버그를 방지합니다.
        if STORY[self.current_idx].get("action") != "CHOICE":
            return

        self.btn_y.setEnabled(False)
        self.btn_n.setEnabled(False)
        self.btn_y.hide()
        self.btn_n.hide()
        self.setFocus(Qt.OtherFocusReason)

        if choice == "Y":
            self.load_node(Y_ENDING_START)
        else:
            self.load_node(N_ENDING_START)

    # ── 이벤트 ──
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.advance()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return):
            self.advance()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 리사이즈 시 현재 노드 다시 렌더링
        if hasattr(self, 'current_idx'):
            node = STORY[self.current_idx]
            self._render_viewport(node.get("sprite"))


# =========================================================================
# WarpCanvas — 유동화 / 자유 선택 편집 캔버스
# =========================================================================
class WarpCanvas(QLabel):
    """유동화 / 자유 선택 편집을 처리하는 캔버스 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mg = parent  # MinigameWidget 참조
        self.start_pos = None
        self.current_pos = None
        self.drag_selection_points = []
        self.brush_radius = 45
        self.setMouseTracking(True)
        self.setFixedSize(CANVAS_W, CANVAS_H)
        self.setCursor(Qt.CrossCursor)

    def _inside_canvas(self, pos):
        x = max(0, min(CANVAS_W - 1, pos.x()))
        y = max(0, min(CANVAS_H - 1, pos.y()))
        return QPoint(x, y)

    def mousePressEvent(self, e):
        if self.mg.phase != "EDIT" or e.button() != Qt.LeftButton:
            return

        pos = self._inside_canvas(e.pos())
        self.start_pos = pos
        self.current_pos = pos

        if self.mg.tool_mode == "SELECT":
            self.drag_selection_points = [pos]
            self.mg.clear_selection(refresh=False)
        self.update()

    def mouseMoveEvent(self, e):
        pos = self._inside_canvas(e.pos())
        self.current_pos = pos

        if self.mg.phase == "EDIT" and self.mg.tool_mode == "SELECT" and self.start_pos is not None:
            if not self.drag_selection_points or pt_dist(
                (self.drag_selection_points[-1].x(), self.drag_selection_points[-1].y()),
                (pos.x(), pos.y())
            ) >= 3:
                self.drag_selection_points.append(pos)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.mg.phase != "EDIT" or not self.start_pos:
            return

        pos = self._inside_canvas(e.pos())
        if e.button() == Qt.LeftButton:
            if self.mg.tool_mode == "LIQUIFY":
                self.mg.apply_warp(self.start_pos, pos, self.brush_radius)
            elif self.mg.tool_mode == "SELECT":
                if len(self.drag_selection_points) >= 3:
                    self.mg.set_selection(self.drag_selection_points)
                else:
                    self.mg.clear_selection()
                self.drag_selection_points = []

        self.start_pos = None
        self.current_pos = None
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.mg.phase != "EDIT":
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 확정된 자유 선택 영역 표시
        if self.mg.selection_points:
            painter.setPen(QPen(QColor(255, 255, 0, 210), 2, Qt.DashLine))
            painter.drawPolygon(QPolygon(self.mg.selection_points))

        # 드래그 중인 자유 선택 영역 표시
        if self.mg.tool_mode == "SELECT" and len(self.drag_selection_points) >= 2:
            painter.setPen(QPen(QColor(255, 255, 0, 170), 2, Qt.DashLine))
            painter.drawPolyline(QPolygon(self.drag_selection_points))

        if not self.current_pos:
            return

        if self.mg.tool_mode == "LIQUIFY":
            if self.start_pos:
                painter.setPen(QPen(QColor(0, 255, 102), 2))
                painter.drawLine(self.start_pos, self.current_pos)
                painter.setPen(QPen(QColor(0, 255, 102, 120), 1))
                painter.drawEllipse(self.start_pos, self.brush_radius, self.brush_radius)
            else:
                painter.setPen(QPen(QColor(0, 255, 255, 80), 1, Qt.DashLine))
                painter.drawEllipse(self.current_pos, self.brush_radius, self.brush_radius)
        elif self.mg.tool_mode == "SELECT":
            painter.setPen(QPen(QColor(255, 255, 0, 130), 1, Qt.DashLine))
            painter.drawText(self.current_pos + QPoint(12, -12), "free select")


# =========================================================================
# Minigame Widget (3단계 독립형 촬영 CV 엔진)
# =========================================================================
class MinigameWidget(QWidget):
    """독립형 촬영 방식의 3단계 얼굴 성형 미니게임."""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.stage = 0        # 0=아직 시작 안함, 1/2/3=각 스테이지
        self.phase = "CAPTURE" # "CAPTURE" or "EDIT"

        # MediaPipe FaceLandmarker (new tasks API)
        self.landmarker = None
        self._init_landmarker()

        self.cap = None
        self.cam_timer = QTimer()
        self.cam_timer.timeout.connect(self._update_webcam)

        self.captured_image = None
        self.current_image = None
        self.current_landmarks = None  # list of (x, y) tuples

        # 편집 도구 상태
        self.tool_mode = "LIQUIFY"  # "LIQUIFY", "SELECT"
        self.selection_points = []
        self.selection_mask = None

        self.target_score = 70
        self.scores = {"exp": 0, "sym": 0, "vln": 0}
        self.init_ui()

    def _init_landmarker(self):
        """FaceLandmarker 초기화 (IMAGE 모드)."""
        if not os.path.exists(MODEL_PATH):
            print(f"[WARNING] 모델 파일 없음: {MODEL_PATH}")
            return
        options = FaceLandmarkerOptions(
            base_options=MpBaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def init_ui(self):
        self.setStyleSheet(f"""
            background-color: {DARK_BG};
            color: {ACCENT_GREEN};
            font-family: '{FONT_MONO}';
        """)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ── 왼쪽: 캔버스 ──
        self.canvas = WarpCanvas(self)
        self.canvas.setStyleSheet(
            f"border: 2px solid {ACCENT_GREEN}; background-color: #000;")
        main_layout.addWidget(self.canvas)

        # ── 오른쪽: 사이드 패널 ──
        side = QVBoxLayout()
        side.setContentsMargins(14, 0, 6, 0)
        side.setSpacing(8)

        title = QLabel("VISION FACE HACKER")
        title.setFont(QFont(FONT_MONO, 16, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT_GREEN};")
        side.addWidget(title)

        self.stage_lbl = QLabel("STAGE 0 / 3  [ READY ]")
        self.stage_lbl.setFont(QFont(FONT_MONO, 11))
        self.stage_lbl.setStyleSheet(f"color: {ACCENT_CYAN};")
        side.addWidget(self.stage_lbl)

        self.guide_lbl = QLabel("마스크 벗고 웹캠 촬영.\n에러 시 데모 로드.")
        self.guide_lbl.setFont(QFont(FONT_MAIN, 10))
        self.guide_lbl.setWordWrap(True)
        self.guide_lbl.setStyleSheet("color: #ccc;")
        side.addWidget(self.guide_lbl)

        self.status_lbl = QLabel("")
        self.status_lbl.setFont(QFont(FONT_MONO, 9))
        self.status_lbl.setStyleSheet("color: #ff3333;")
        side.addWidget(self.status_lbl)

        side.addStretch()

        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Score: %p%")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {ACCENT_GREEN};
                border-radius: 4px;
                text-align: center;
                color: #fff;
                background: #111;
                height: 22px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ACCENT_GREEN}, stop:1 {ACCENT_CYAN});
                border-radius: 3px;
            }}
        """)
        side.addWidget(self.progress_bar)

        # 스코어 라벨
        self.score_lbl = QLabel("현재 단계 점수: --\n목표: 70")
        self.score_lbl.setFont(QFont(FONT_MONO, 10))
        self.score_lbl.setStyleSheet(
            "background-color: #1a1a24; padding: 10px; border-radius: 4px;")
        side.addWidget(self.score_lbl)

        # ── 편집 도구 패널 ──
        tool_title = QLabel("TOOLS")
        tool_title.setFont(QFont(FONT_MONO, 10, QFont.Bold))
        tool_title.setStyleSheet(f"color: {ACCENT_YELLOW}; margin-top: 4px;")
        side.addWidget(tool_title)

        tool_style = f"""
            QPushButton {{
                background-color: #151521;
                color: {ACCENT_YELLOW};
                border: 1px solid {ACCENT_YELLOW};
                border-radius: 4px;
                padding: 8px 6px;
                font-size: 9pt;
                font-family: '{FONT_MONO}';
            }}
            QPushButton:checked {{
                background-color: {ACCENT_YELLOW};
                color: #000;
            }}
            QPushButton:disabled {{
                border-color: #333;
                color: #444;
            }}
        """
        tool_row = QHBoxLayout()
        tool_row.setSpacing(5)
        self.btn_tool_liquify = QPushButton("유동화")
        self.btn_tool_select = QPushButton("자유선택")
        for btn, mode in (
            (self.btn_tool_liquify, "LIQUIFY"),
            (self.btn_tool_select, "SELECT"),
        ):
            btn.setCheckable(True)
            btn.setStyleSheet(tool_style)
            btn.clicked.connect(lambda checked, m=mode: self.set_tool_mode(m))
            tool_row.addWidget(btn)
        side.addLayout(tool_row)

        select_row = QHBoxLayout()
        select_row.setSpacing(5)
        self.btn_zoom_plus = QPushButton("선택 +")
        self.btn_zoom_plus.setStyleSheet(tool_style)
        self.btn_zoom_plus.clicked.connect(self.enlarge_selection)
        select_row.addWidget(self.btn_zoom_plus)
        side.addLayout(select_row)

        side.addStretch()

        # ── 버튼들 ──
        btn_base = f"""
            QPushButton {{
                background-color: #1c1c28;
                border: 2px solid {ACCENT_GREEN};
                color: {ACCENT_GREEN};
                padding: 10px;
                font-weight: bold;
                font-family: '{FONT_MONO}';
                font-size: 10pt;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_GREEN};
                color: #000;
            }}
            QPushButton:disabled {{
                border-color: #333;
                color: #444;
            }}
        """

        self.btn_capture = QPushButton("📷  CAPTURE FACE")
        self.btn_capture.setStyleSheet(btn_base)
        self.btn_capture.clicked.connect(self.on_capture)
        side.addWidget(self.btn_capture)

        self.btn_demo = QPushButton("🖼  USE DEMO FACE")
        self.btn_demo.setStyleSheet(btn_base)
        self.btn_demo.clicked.connect(self.load_demo)
        side.addWidget(self.btn_demo)

        self.btn_reset = QPushButton("↺  RESET")
        self.btn_reset.setStyleSheet(btn_base)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_reset.setEnabled(False)
        side.addWidget(self.btn_reset)

        self.btn_retake = QPushButton("📷  다시 촬영")
        self.btn_retake.setStyleSheet(btn_base)
        self.btn_retake.clicked.connect(self.on_retake)
        self.btn_retake.setEnabled(False)
        side.addWidget(self.btn_retake)

        btn_next_style = f"""
            QPushButton {{
                background-color: #1c1c28;
                border: 2px solid {ACCENT_CYAN};
                color: {ACCENT_CYAN};
                padding: 10px;
                font-weight: bold;
                font-family: '{FONT_MONO}';
                font-size: 10pt;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_CYAN};
                color: #000;
            }}
            QPushButton:disabled {{
                border-color: #333;
                color: #444;
            }}
        """
        self.btn_next = QPushButton("▶  NEXT STAGE")
        self.btn_next.setStyleSheet(btn_next_style)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_next.setEnabled(False)
        side.addWidget(self.btn_next)

        btn_fail_style = btn_base.replace(ACCENT_GREEN, "#ff4444")
        self.btn_giveup = QPushButton("✕  GIVE UP")
        self.btn_giveup.setStyleSheet(btn_fail_style)
        self.btn_giveup.clicked.connect(self.on_giveup)
        side.addWidget(self.btn_giveup)

        main_layout.addLayout(side)

        self.set_tool_mode("LIQUIFY")
        self._set_edit_controls_enabled(False)

    def set_tool_mode(self, mode):
        """편집 도구 전환: 유동화 / 자유 선택."""
        self.tool_mode = mode
        if hasattr(self, "btn_tool_liquify"):
            self.btn_tool_liquify.setChecked(mode == "LIQUIFY")
            self.btn_tool_select.setChecked(mode == "SELECT")

        if mode == "LIQUIFY":
            self.canvas.setCursor(Qt.CrossCursor)
            self.status_lbl.setText("도구: 유동화 — 드래그 방향으로 픽셀을 밀어냅니다.")
        elif mode == "SELECT":
            self.canvas.setCursor(Qt.CrossCursor)
            self.status_lbl.setText("도구: 자유 선택 — 드래그로 영역을 따고 [선택 +]로 확대합니다.")
        self.canvas.update()

    def _set_edit_controls_enabled(self, enabled):
        controls = [
            self.btn_tool_liquify, self.btn_tool_select, self.btn_zoom_plus
        ]
        for w in controls:
            w.setEnabled(enabled)

    # ── 활성화 (미니게임 진입) ──
    def activate(self):
        self.stage = 1
        self.phase = "CAPTURE"
        self._enter_capture_phase()

    def _enter_capture_phase(self):
        """캡처 위상 진입 — 웹캠 열기 시도."""
        self.phase = "CAPTURE"
        self.captured_image = None
        self.current_image = None
        self.current_landmarks = None
        self.progress_bar.setValue(0)
        self.btn_next.setEnabled(False)
        self.btn_next.setText("▶  NEXT STAGE")
        self.btn_reset.setEnabled(False)
        self.btn_retake.setEnabled(False)
        self.btn_capture.show()
        self.btn_capture.setEnabled(False)
        self.clear_selection(refresh=False)
        self._set_edit_controls_enabled(False)
        self.set_tool_mode("LIQUIFY")

        stage_names = {1: "EXPRESSION", 2: "SYMMETRY", 3: "V-LINE"}
        self.stage_lbl.setText(
            f"STAGE {self.stage} / 3  [ {stage_names.get(self.stage, '?')} ]")

        guides = {
            1: "📷 웹캠으로 얼굴을 촬영하세요.\n1단계는 표정/눈 점수만 평가합니다.\n(목표: 70점)",
            2: "📷 웹캠으로 얼굴을 새로 촬영하세요.\n2단계는 좌우 대칭 점수만 평가합니다.\n(목표: 70점)",
            3: "📷 웹캠으로 얼굴을 새로 촬영하세요.\n3단계는 V라인 점수만 평가합니다.\n(목표: 70점)",
        }
        self.guide_lbl.setText(guides.get(self.stage, ""))
        self.score_lbl.setText("현재 단계 점수: --\n목표: 70")
        self.status_lbl.setText("")

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cam_timer.start(33)  # ~30fps
            self.btn_demo.hide()
        else:
            self.btn_capture.setEnabled(False)
            self.btn_demo.show()
            self.status_lbl.setText("웹캠 미감지 — 데모 얼굴 사용")
            # 검은 화면 표시
            blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            cv2.putText(blank, "No Webcam Detected", (220, CANVAS_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 102), 2)
            self._display_cv(blank)

    def _stop_cam(self):
        self.cam_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    # ── 웹캠 업데이트 ──
    def _update_webcam(self):
        if self.phase != "CAPTURE" or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(cv2.flip(frame, 1), (CANVAS_W, CANVAS_H))
        self.captured_image = frame.copy()

        # 얼굴 인식 (new tasks API)
        landmarks = self._detect_from_frame(frame)
        if landmarks:
            self.current_landmarks = landmarks
            self.btn_capture.setEnabled(True)
            self.status_lbl.setText("")
            # 얼굴 메쉬 포인트 그리기
            for pt in self.current_landmarks[::3]:
                cv2.circle(frame, pt, 1, (0, 255, 102), -1)
        else:
            self.current_landmarks = None
            self.btn_capture.setEnabled(False)
            self.status_lbl.setText("얼굴 미인식")
        self._display_cv(frame)

    # ── 데모 얼굴 로드 ──
    def load_demo(self):
        """
        데모 얼굴을 로드합니다.
        images/demo_face.png가 있으면 사용하고, 없으면 사용자가 바로 알 수 있도록
        캔버스에 안내 문구를 띄웁니다. 실제 점수 계산에는 MediaPipe가 인식할 수 있는
        정면 얼굴 사진이 필요합니다.
        """
        demo_path = get_img_path("demo_face.png")
        if not demo_path:
            self.status_lbl.setText("[ERROR] images/demo_face.png 없음 — 웹캠을 쓰거나 demo_face.png를 추가하세요.")
            blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            lines = [
                "demo_face.png not found",
                "Add a front-facing face photo to images/demo_face.png",
                "or use webcam capture."
            ]
            y0 = CANVAS_H // 2 - 55
            for i, line in enumerate(lines):
                cv2.putText(blank, line, (90, y0 + i * 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 102), 2, cv2.LINE_AA)
            self._display_cv(blank)
            return

        img = cv2.imread(demo_path)
        if img is None:
            self.status_lbl.setText("[ERROR] demo_face.png를 읽을 수 없습니다.")
            return

        img = cv2.resize(img, (CANVAS_W, CANVAS_H))
        self._stop_cam()
        self.captured_image = img.copy()
        self.current_image = img.copy()
        self.btn_demo.hide()
        self.btn_capture.hide()
        self.btn_reset.setEnabled(True)
        self.btn_retake.setEnabled(True)
        self.phase = "EDIT"
        self._set_edit_controls_enabled(True)
        self._detect_landmarks()
        self._update_scores()
        self._display_with_overlay()
        self._update_guide_for_edit()

    # ── 촬영 ──
    def on_capture(self):
        if self.phase == "CAPTURE" and self.current_landmarks:
            self._stop_cam()
            self.current_image = self.captured_image.copy()
            self.btn_capture.hide()
            self.btn_demo.hide()
            self.btn_reset.setEnabled(True)
            self.btn_retake.setEnabled(True)
            self.phase = "EDIT"
            self._set_edit_controls_enabled(True)
            self._detect_landmarks()
            self._update_scores()
            self._display_with_overlay()
            self._update_guide_for_edit()

    # ── 리셋 (현재 스테이지의 원본으로 되돌림) ──
    def on_reset(self):
        if self.captured_image is not None:
            self.current_image = self.captured_image.copy()
            self.clear_selection(refresh=False)
            self._detect_landmarks()
            self._update_scores()
            self._display_with_overlay()

    # ── 다시 촬영 (현재 스테이지에서 새 사진 찍기) ──
    def on_retake(self):
        """RESET은 현재 촬영본으로 되돌리고, 다시 촬영은 현재 스테이지의 캡처 단계로 돌아갑니다."""
        self._stop_cam()
        self.clear_selection(refresh=False)
        self._enter_capture_phase()

    # ── 다음 스테이지 ──
    def on_next(self):
        if self.stage < 3:
            self.stage += 1
            self._stop_cam()
            self._enter_capture_phase()
        else:
            # 3스테이지 완료
            self._stop_cam()
            self.parent_app.finish_minigame(True)

    # ── 포기 ──
    def on_giveup(self):
        self._stop_cam()
        self.parent_app.finish_minigame(False)

    # ── 유동화 적용 ──
    def apply_warp(self, p1, p2, rad):
        if self.current_image is None:
            return
        self.current_image = warp_image(
            self.current_image, p1.x(), p1.y(), p2.x(), p2.y(), rad)
        self._detect_landmarks()
        self._update_scores()
        self._display_with_overlay()

    # ── 자유 선택 + 확대 ──
    def clear_selection(self, refresh=True):
        self.selection_points = []
        self.selection_mask = None
        if refresh:
            self._display_with_overlay()
            self.canvas.update()

    def set_selection(self, points):
        if self.current_image is None or len(points) < 3:
            self.clear_selection()
            return
        self.selection_points = [QPoint(p.x(), p.y()) for p in points]
        poly = np.array([[p.x(), p.y()] for p in self.selection_points], dtype=np.int32)
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        self.selection_mask = mask
        self.status_lbl.setText("자유 선택 완료 — [선택 +] 버튼으로 선택 영역을 확대할 수 있습니다.")
        self._display_with_overlay()
        self.canvas.update()

    def enlarge_selection(self):
        """자유 선택 영역을 Paint의 선택 확대처럼 중심 기준으로 키워 합성합니다."""
        if self.current_image is None or self.selection_mask is None:
            self.status_lbl.setText("먼저 [자유선택] 도구로 확대할 영역을 드래그하세요.")
            return

        mask = self.selection_mask
        x, y, w, h = cv2.boundingRect(mask)
        if w <= 2 or h <= 2:
            self.status_lbl.setText("선택 영역이 너무 작습니다.")
            return

        src = self.current_image[y:y + h, x:x + w].copy()
        src_mask = mask[y:y + h, x:x + w].copy()
        new_w = max(2, int(w * SELECTION_SCALE_STEP))
        new_h = max(2, int(h * SELECTION_SCALE_STEP))
        resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        resized_mask = cv2.resize(src_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        cx = x + w // 2
        cy = y + h // 2
        nx = cx - new_w // 2
        ny = cy - new_h // 2

        # 캔버스 밖으로 나가는 부분 클리핑
        dst_x1 = max(0, nx)
        dst_y1 = max(0, ny)
        dst_x2 = min(CANVAS_W, nx + new_w)
        dst_y2 = min(CANVAS_H, ny + new_h)
        src_x1 = dst_x1 - nx
        src_y1 = dst_y1 - ny
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            self.status_lbl.setText("확대 결과가 화면 밖에 있습니다.")
            return

        patch = resized[src_y1:src_y2, src_x1:src_x2]
        patch_mask = resized_mask[src_y1:src_y2, src_x1:src_x2].astype(np.float32) / 255.0
        patch_mask = patch_mask[..., None]
        roi = self.current_image[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        blended = patch.astype(np.float32) * patch_mask + roi * (1.0 - patch_mask)
        self.current_image[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

        # 확대 후에도 선택 상태를 유지해서 + 버튼을 연속으로 누를 수 있게 함
        new_mask = np.zeros_like(self.selection_mask)
        new_mask[dst_y1:dst_y2, dst_x1:dst_x2] = (patch_mask[..., 0] * 255).astype(np.uint8)
        self.selection_mask = new_mask
        self.selection_points = [
            QPoint(dst_x1, dst_y1), QPoint(dst_x2, dst_y1),
            QPoint(dst_x2, dst_y2), QPoint(dst_x1, dst_y2)
        ]

        self._detect_landmarks()
        self._update_scores()
        self._display_with_overlay()
        self.status_lbl.setText("선택 영역을 18% 확대했습니다. +를 다시 누르면 계속 커집니다.")

    # ── 랜드마크 검출 ──
    def _detect_from_frame(self, frame):
        """프레임에서 얼굴 랜드마크를 검출하여 (x,y) 좌표 리스트를 반환."""
        if self.landmarker is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if result.face_landmarks and len(result.face_landmarks) > 0:
            h, w = frame.shape[:2]
            return [(int(lm.x * w), int(lm.y * h))
                    for lm in result.face_landmarks[0]]
        return None

    def _detect_landmarks(self):
        if self.current_image is None:
            return
        landmarks = self._detect_from_frame(self.current_image)
        if landmarks:
            self.current_landmarks = landmarks
            self.status_lbl.setText("")
        else:
            self.current_landmarks = None
            self.status_lbl.setText("[ERROR] 얼굴 인식 실패. Reset 권장.")

    # ── 점수 계산 ──
    def _update_scores(self):
        if not self.current_landmarks:
            self.btn_next.setEnabled(False)
            self.score_lbl.setText("현재 단계 점수: --\n목표: 70")
            return

        pts = self.current_landmarks

        # Expression score (눈 개방도 + 입꼬리)
        e_open = (
            (pt_dist(pts[159], pts[145]) / max(1, pt_dist(pts[33], pts[133]))) +
            (pt_dist(pts[386], pts[374]) / max(1, pt_dist(pts[263], pts[362])))
        ) / 2
        smile = (
            ((pts[0][1] + pts[17][1]) / 2) -
            ((pts[61][1] + pts[291][1]) / 2)
        ) / max(1, pt_dist(pts[61], pts[291]))
        sc_exp = min(100, max(0,
            40 + np.clip((e_open - 0.15) / 0.15, 0, 1) * 30 +
            np.clip((smile + 0.04) / 0.2, 0, 1) * 30))

        # Symmetry score
        cx = pts[1][0]
        pairs = [(33, 263), (70, 300), (61, 291), (234, 454)]
        sym_err = sum(
            (abs((cx - pts[l][0]) - (pts[r][0] - cx)) +
             abs(pts[l][1] - pts[r][1])) /
            max(1, pt_dist(pts[234], pts[454]))
            for l, r in pairs
        )
        sc_sym = min(100, max(0, 100 - sym_err * 160))

        # V-Line score (광대 대비 턱 폭, 턱 끝 중앙 정렬, 턱 길이)
        cheek_w = max(1, pt_dist(pts[234], pts[454]))
        jaw_w = pt_dist(pts[172], pts[397])
        cx_face = (pts[234][0] + pts[454][0]) / 2
        mouth_y = (pts[13][1] + pts[14][1]) / 2

        jaw_ratio = jaw_w / cheek_w                 # 낮을수록 갸름함
        chin_len = max(0, (pts[152][1] - mouth_y) / cheek_w)
        chin_center_err = abs(pts[152][0] - cx_face) / cheek_w

        v_err = (
            abs(jaw_ratio - 0.64) / 0.28 +
            abs(chin_len - 0.34) / 0.22 +
            chin_center_err * 3.0
        ) / 3
        sc_vln = min(100, max(0, 100 - v_err * 115))

        self.scores = {"exp": sc_exp, "sym": sc_sym, "vln": sc_vln}

        # 단계별 독립 평가: 현재 단계의 점수만 진행 조건에 사용합니다.
        target = self.target_score
        if self.stage == 1:
            current_score = sc_exp
            score_name = "Expression"
        elif self.stage == 2:
            current_score = sc_sym
            score_name = "Symmetry"
        else:  # stage 3
            current_score = sc_vln
            score_name = "V-Line"

        self.progress_bar.setValue(int(current_score))
        passed = current_score >= target
        self.btn_next.setEnabled(passed)

        if self.stage == 3 and passed:
            self.btn_next.setText("✔  COMPLETE")
        elif self.stage == 3:
            self.btn_next.setText("▶  NEXT STAGE")

        self.score_lbl.setText(
            f"{score_name}: {current_score:.1f}\n목표: {target}\n※ 이 단계는 {score_name}만 평가")

    def _update_guide_for_edit(self):
        guides = {
            1: "유동화로 눈과 표정을 조정하세요.\n1단계는 Expression 점수만 봅니다.\n(목표: Expression ≥ 70)",
            2: "유동화/자유선택으로 좌우 위치를 맞추세요.\n2단계는 Symmetry 점수만 봅니다.\n(목표: Symmetry ≥ 70)",
            3: "유동화로 턱선을 안쪽으로 밀고, 자유선택 +로 턱/입 주변을 조정하세요.\n3단계는 V-Line 점수만 봅니다.\n(목표: V-Line ≥ 70)",
        }
        self.guide_lbl.setText(guides.get(self.stage, ""))

    # ── 오버레이 표시 ──
    def _display_with_overlay(self):
        if self.current_image is None:
            return
        img = self.current_image.copy()
        if self.current_landmarks:
            pts = self.current_landmarks
            if self.stage == 1:
                # 눈 주변 가이드 타원
                cv2.ellipse(img, pts[159], (28, 16), 0, 0, 360,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.ellipse(img, pts[386], (28, 16), 0, 0, 360,
                            (0, 255, 255), 1, cv2.LINE_AA)
            elif self.stage == 2:
                # 중앙 대칭선
                color = (0, 255, 102) if self.scores["sym"] >= self.target_score else (0, 0, 255)
                cv2.line(img, pts[10], pts[152], color, 2)
            elif self.stage == 3:
                # V라인 가이드: 광대→턱선→턱끝을 삼각 라인으로 표시
                v_pts = np.array([
                    pts[234], pts[172], pts[152], pts[397], pts[454]
                ], dtype=np.int32)
                color = (0, 255, 102) if self.scores.get("vln", 0) >= self.target_score else (0, 180, 255)
                cv2.polylines(img, [v_pts], False, color, 2, cv2.LINE_AA)

                cx = int((pts[234][0] + pts[454][0]) / 2)
                cv2.line(img, (cx, pts[10][1]), (cx, pts[152][1]), (255, 255, 0), 1, cv2.LINE_AA)
                # 턱 양쪽을 안쪽으로 당기라는 시각 가이드
                cv2.arrowedLine(img, pts[172], (int(cx - 35), pts[172][1] + 20),
                                (255, 180, 0), 2, cv2.LINE_AA, tipLength=0.25)
                cv2.arrowedLine(img, pts[397], (int(cx + 35), pts[397][1] + 20),
                                (255, 180, 0), 2, cv2.LINE_AA, tipLength=0.25)
        self._display_cv(img)

    # ── OpenCV → QPixmap 변환 표시 ──
    def _display_cv(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        self.canvas.setPixmap(QPixmap.fromImage(qimg))
        self.canvas.update()


# =========================================================================
# Title Widget (타이틀 화면)
# =========================================================================
class TitleWidget(QWidget):
    """게임 시작 전 타이틀 화면."""

    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(f"background-color: {DARK_BG};")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # 타이틀 텍스트
        title = QLabel("VISION FACE HACKER")
        title.setFont(QFont(FONT_MONO, 32, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT_GREEN};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("V-Line Face Simulator")
        subtitle.setFont(QFont(FONT_MONO, 14))
        subtitle.setStyleSheet(f"color: {ACCENT_CYAN};")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        desc = QLabel("이미지 비전 기술을 활용한 비주얼 노벨")
        desc.setFont(QFont(FONT_MAIN, 11))
        desc.setStyleSheet("color: #888;")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        layout.addSpacing(40)

        # START 버튼
        btn_start = QPushButton("▶  START GAME")
        btn_start.setFont(QFont(FONT_MONO, 14, QFont.Bold))
        btn_start.setFixedSize(300, 55)
        btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: #1a1a28;
                color: {ACCENT_GREEN};
                border: 2px solid {ACCENT_GREEN};
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_GREEN};
                color: #000;
            }}
        """)
        btn_start.clicked.connect(self.parent_app.start_game)
        layout.addWidget(btn_start, alignment=Qt.AlignCenter)

        layout.addSpacing(20)

        credit = QLabel("Image Vision Term Project  |  PyQt5 + OpenCV + MediaPipe")
        credit.setFont(QFont(FONT_MONO, 9))
        credit.setStyleSheet("color: #444;")
        credit.setAlignment(Qt.AlignCenter)
        layout.addWidget(credit)


# =========================================================================
# Main Application Window
# =========================================================================
class FaceHackerMainWindow(QMainWindow):
    """메인 윈도우 — QStackedWidget으로 화면 전환을 관리합니다."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Face Hacker — V-Line Face Simulator")
        # 미니게임 캔버스와 도구 패널이 답답하지 않도록 기본 UI를 확장
        self.setMinimumSize(1240, 820)
        self.resize(1320, 860)

        # 다크 팔레트
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(DARK_BG))
        palette.setColor(QPalette.WindowText, QColor("#e0e0e0"))
        self.setPalette(palette)

        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)

        # 위젯 생성
        self.title_widget = TitleWidget(self)
        self.story_widget = StoryWidget(self)
        self.minigame_widget = MinigameWidget(self)

        self.stacked.addWidget(self.title_widget)      # index 0
        self.stacked.addWidget(self.story_widget)       # index 1
        self.stacked.addWidget(self.minigame_widget)    # index 2

        self.stacked.setCurrentIndex(0)

    def show_title(self):
        self.stacked.setCurrentIndex(0)

    def start_game(self):
        self.stacked.setCurrentIndex(1)
        self.story_widget.load_node(0)
        self.story_widget.setFocus()

    def start_minigame(self):
        self.minigame_widget.activate()
        self.stacked.setCurrentIndex(2)

    def finish_minigame(self, success):
        self.stacked.setCurrentIndex(1)
        if success:
            self.story_widget.load_node(MINIGAME_SUCCESS_NODE)
        else:
            self.story_widget.load_node(MINIGAME_FAIL_NODE)
        self.story_widget.setFocus()


# =========================================================================
# Entry Point
# =========================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FaceHackerMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
