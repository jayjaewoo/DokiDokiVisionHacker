# DokiDokiVisionHacker

철학적 비주얼 노벨(Visual Novel) + 얼굴 편집 미니게임

코로나 시기 마스크 사회를 배경으로, 외모와 욕망, 이미지와 실재의 관계를 다루는 비주얼 노벨 프로젝트.

플레이어는 얼굴 편집 미니게임을 통해 자신의 얼굴을 수정한 뒤, 주인공 희지와의 관계 속에서 두 가지 엔딩에 도달하게 된다.

---

## 플레이 정보

**주의) 설치한 폴더에 한글이 있을 경우 실행이 되지 않습니다.**

* 장르: 비주얼 노벨 / 얼굴 편집 미니게임
* 플레이 타임: 약 5~10분
* 엔딩 수: 3개
* 조작 방식: 마우스 클릭, 키보드 Space/Enter
* 필요 장비: 웹캠 필수

---

## 특징

### 비주얼 노벨 엔진

* PyQt5 기반 비주얼 노벨 UI
* 배경 이미지 및 캐릭터 스프라이트 지원
* 타이핑 애니메이션
* 선택지 분기 시스템
* 다중 엔딩

### 얼굴 편집 미니게임

* OpenCV 기반 픽셀 유동화
* MediaPipe Face Landmarker 사용
* 3단계 미션 구조

#### STAGE 1 : Expression

눈과 표정 보정

목표 점수 : 70점 이상

#### STAGE 2 : Symmetry

얼굴 좌우 대칭 보정

목표 점수 : 70점 이상

#### STAGE 3 : V-Line

턱선과 얼굴형 보정

목표 점수 : 70점 이상

### 편집 기능

* 픽셀 유동화
* 자유 선택
* 선택 영역 확대
* RESET
* 다시 촬영

---

## 엔딩

플레이어의 선택에 따라 서로 다른 결말에 도달하게 됩니다.

어떤 결말은 그 기회조차 앗아가고,
어떤 결말은 진실을 마주하게 만들고,
어떤 결말은 끝내 진실에 닿지 못하게 만듭니다.

당신이 선택한 얼굴과 선택한 관계는
어떤 이야기를 완성하게 될까요?

총 3개의 엔딩이 준비되어 있습니다.

---

### 스크린샷

![Opening](screenshot1.png)

![Camera](screenshot2.png)

---

## 프로젝트 구조

```text
VisionFaceHacker/
│
├─ main.py
├─ run_game.bat
├─ requirements.txt
├─ face_landmarker.task
│
└─ images/
   ├─ classroom_bg.png
   ├─ cafe_bg.png
   ├─ heeji_normal.png
   ├─ heeji_excited.png
   ├─ heeji_sad.png
   ├─ orphee.png
   ├─ les_amants.png
   └─ demo_face.png
```

---

## 설치

Python 3.10 이상 권장

```bash
pip install -r requirements.txt
```

---

## 실행

```bash
python main.py
```

또는

```text
run_game.bat 더블클릭
```

---

## 개발 환경

* Python
* PyQt5
* OpenCV
* NumPy
* MediaPipe

---

## 라이선스

개인 학습 및 포트폴리오 용도.

사용된 철학 텍스트, 미술 작품, 이미지의 저작권은 각 권리자에게 있습니다.

---

## 제작 의도

20대 청년들에게 익숙한 미연시의 문법을 통해,
실재와 허구가 뒤섞인 사회 속에서 외모가 어떤 위치를 차지하는지 이야기하고 싶었습니다.

우리는 타인의 얼굴을 보지만,
실제로 사랑하는 것은 얼굴 그 자체일 수도,
그 얼굴 위에 덧씌운 이미지일 수도 있습니다.

이 작품은 그런 질문을 플레이어 스스로 경험해볼 수 있도록 구성했습니다.

"사랑받는 것은 진짜 나인가,
아니면 내가 만들어낸 이미지인가?"
