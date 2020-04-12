# 프로젝트 소개

1. 프로젝트 명 : ColdEyes Project

2. 팀원: 박경민,최승진,신호준,이가원,이진재

3. 프로젝트 소개: 영상처리와 딥러닝 기반한 실시간화재 감지시스템

4. 개발기간: 2019.07~2019.08

5. 데모영상: https://www.youtube.com/watch?v=8OwfcMeB8NY

5. 결과화면

<b>실행화면 1) 페이지 UI구성 </b>
<img src="https://user-images.githubusercontent.com/37204852/78981092-29fde680-7b5a-11ea-80db-fe15bf4f79d4.png">

<b>실행화면 2) 화재 감지 시 알림</b>
<img src="https://user-images.githubusercontent.com/37204852/78981312-9ed12080-7b5a-11ea-9d35-3ad8531701ef.png">

# 실행방법

윈도우 환경에서 작성되었습니다.

1. 아나콘다3 파이썬, Opecv, Tensorflow, Pytorch가 설치되어 있어야합니다.

2. CUDA지원 GPU를 가지고 있으면, 쾌적한 실행을 위해 CUDA 설치를 추천드립니다.

4. ROOT 경로에 yolov3.weights(학습모델) 파일이 존재해야합니다.

5. 학습모델 파일은 용량이 커 첨부하지 못했으므로, 필요하신분은 qkrrudals689@naver.com에 메일부탁드립니다.

6. py manage.py runserver -> http://127.0.0.1:8000/ 으로 접속하시면 됩니다.
