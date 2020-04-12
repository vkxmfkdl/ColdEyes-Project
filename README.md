# 프로젝트 소개

1. 프로젝트 명 : ColdEyes Project

2. 팀원: 박경민,최승진,신호준,이가원,이진재

3. 프로젝트 소개: 영상처리와 딥러닝 기반한 실시간화재 감지시스템

4. 개발기간: 2019.08~2019.09

5. 데모영상: https://www.youtube.com/watch?v=8OwfcMeB8NY

5. 결과화면

<b>실행화면 1) 페이지 UI구성 </b>
<img src="https://user-images.githubusercontent.com/37204852/78981092-29fde680-7b5a-11ea-80db-fe15bf4f79d4.png">

<b>실행화면 2) 화재 감지 시 알림</b>
<img src="https://user-images.githubusercontent.com/37204852/78981312-9ed12080-7b5a-11ea-9d35-3ad8531701ef.png">

# 구글 Cloud 음성인식 API

https://webnautes.tistory.com/1247 참조

# 실행방법

윈도우 환경에서 작성되었습니다.

1. 아나콘다3 파이썬이 설치되어 있어야합니다.

2. `conda env create -f environment.yml` 명령어로 가상환경을 설치합니다

3. `conda activate voiceenv` 명령어로 가상환경을 실행시킵니다.

4. 음성인식을 구동시키려면 개인 컴퓨터에 구글 음성인식 api가 설치되어 있어야합니다.

5. 음성분석을 하기 위해서는 uploadproject 폴더로 들어가 `python manage.py runserver` 명령어를 실행하면 됩니다.

6. 웹 페이지 접속 기본 url은 localhost:8000 입니다. 이는 실행하는 사람의 환경에 따라 다를 수도 있습니다.
