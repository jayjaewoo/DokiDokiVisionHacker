# 임희지 쨩이 쉽게 속을 리 없잖아 무리무리~! (※무리가 아니었다?!)
컴퓨터 비전 기술(MediaPipe, OpenCV, EasyOCR)을 활용한 인터랙티브 비주얼 노벨 및 이미지 비전 기술 훈련 시스템 구현 프로젝트입니다.

---

## 1. 프로젝트 소개
본 프로젝트는 이미지 비전 기술을 대중적으로 친숙한 `미소녀 연애 시뮬레이션(미연시)` 클리셰 및 게임 메커니즘에 융합한 텀프로젝트입니다. 미연시 구조 안에서 주인공이 상대방의 인지 시스템(인공지능 모델 및 인간의 인지 필터)을 기만하기 위해 다양한 컴퓨터 비전 기술을 직접 제어하는 과정을 제공합니다.

### 기획 배경 및 의도
 * **비전 원리의 게임화**: 전통적인 디지털 이미지 처리(Digital Image Processing) 기법과 머신러닝 모델을 미니게임 형태의 퍼즐로 설계하여 기술적 직관을 시각적으로 제공합니다.
 * **직관적인 파라미터 제어**: 복잡한 비전 알고리즘의 변수를 슬라이더 및 드래그 조작계로 매핑하여 실시간 픽셀 변화와 모델 추론 결과 변화를 한눈에 볼 수 있도록 설계했습니다.
 * **하이브리드 엔진 설계**: 연출에 최적화된 비주얼 노블 엔진인 Ren'Py와 비전 알고리즘 파이프라인을 담당하는 PyQt5 기반 독립 프로세스를 서브프로세스 인터페이스로 연동했습니다.

---

## 2. 시스템 아키텍처 및 기술 스택
```
[Ren'Py Engine]
  - 스토리라인 및 캐릭터 연출
  - subprocess 모듈을 통한 외부 프로그램 호출
       │
       ▼ (실행)
[PyQt5 Vision Engine (minigame.py)]
  - PyQt5 UI / 웹캠 프레임 렌더링
  - MediaPipe, OpenCV, EasyOCR 파이프라인 연산
       │
       ▼ (결과 기록: result.txt)
[Ren'Py Engine 복귀]
  - 결과 파싱 후 연출 분기 및 호감도 트리거 반영

```
### 기술 스택 (Tech Stack)
 * **Game Engine**: Ren'Py (Python 기반 비주얼 노블 엔진)
 * **GUI Framework**: PyQt5
 * **Libraries**: OpenCV (v4.x), MediaPipe (Face Mesh), EasyOCR, NumPy
 * **Language**: Python 3.10+

---

## 3. 핵심 스테이지 스펙

### Stage 1: 외모 해킹 (Facial Landmark Matching)
 * **기술 스택**: MediaPipe Face Mesh (468 랜드마크 추출), OpenCV 좌표 거리 연산
 * **구현 핵심**: 노트북 웹캠을 연동하여 사용자의 실시간 주요 얼굴 좌표(눈, 코, 입 등)를 검출합니다. 화면에 표시된 타겟 얼굴(이상형 황금비율) 가이드라인과의 유클리디안 거리(L2 \text{ Distance}) 오차율이 임계치 이하가 되도록 조작 및 동기화합니다.

### Stage 2: 지식 해킹 (Denoising & OCR Decoding)
 * **기술 스택**: OpenCV 필터링 및 커널 연산, EasyOCR 문자 탐지 엔진
 * **구현 핵심**: 심한 가우시안 블러 및 임의 노이즈가 주입된 고서적 이미지에 가변 커널 필터를 적용합니다. 사용자는 슬라이더를 통해 디노이징 커널 크기 및 라플라시안 샤프닝 가중치를 조절하여 텍스트 가독성을 복원하고, 복원 완료 시 OCR을 호출해 정답 문자열을 파싱해 냅니다.

### Stage 3: 재력 해킹 (Contrast/Brightness Transformation)
 * **기술 스택**: Brightness & Contrast Mapping, 픽셀 마스킹
 * **구현 핵심**: 싸구려 모조 반지의 명암비(\alpha)와 밝기(\beta) 변수를 선형 변환 수식 g(x,y) = \alpha \cdot f(x,y) + \beta를 기반으로 실시간 튜닝합니다. 최적의 반사도 조건을 충족하는 순간 특정 마스크 영역(큐빅)에 광채 오버레이를 생성하여 보석 분류 시스템의 임계치를 우회합니다.

---

## 4. 빌드 및 테스트 가이드
### 개발 환경 설치
```bash
# 가상환경 격리 및 활성화
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 종속성 라이브러리 설치
pip install PyQt5 opencv-python mediapipe easyocr numpy

```
### 실행 프로토콜
 1. 본 레포지토리를 클론합니다.
 2. 렌파이(Ren'Py) SDK에서 해당 프로젝트 경로를 지정하여 런처로 실행합니다.
 3. 렌파이 대사 진행 중 미니게임 트리거에 도달하면 외부 PyQt5 프로그램(minigame.py)이 자동 기동됩니다.
