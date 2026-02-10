# 가상환경 pip 설치 가이드

가상환경이 Python 3.12.10으로 생성되었지만 pip가 설치되지 않은 경우, 다음 방법으로 pip를 설치할 수 있습니다.

## 방법 1: get-pip.py 사용 (권장)

1. **get-pip.py 다운로드**:
   ```powershell
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   ```
   또는 브라우저에서 직접 다운로드:
   https://bootstrap.pypa.io/get-pip.py

2. **pip 설치**:
   ```powershell
   .\venv\Scripts\python.exe get-pip.py
   ```

3. **설치 확인**:
   ```powershell
   .\venv\Scripts\python.exe -m pip --version
   ```

4. **임시 파일 삭제**:
   ```powershell
   Remove-Item get-pip.py
   ```

## 방법 2: 시스템 Python의 pip 사용

시스템에 설치된 Python 3.12의 pip를 사용하여 가상환경에 pip 설치:

```powershell
# 시스템 Python 3.12의 pip로 가상환경에 pip 설치
py -3.12 -m pip install --target .\venv\Lib\site-packages pip
```

## 방법 3: setup_python312.bat 실행

제공된 스크립트를 실행하면 자동으로 처리됩니다:

```powershell
.\setup_python312.bat
```

## pip 설치 후 패키지 설치

pip가 설치되면 다음 명령어로 패키지를 설치하세요:

```powershell
# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 패키지 설치
pip install -r requirements.txt
```

또는 가상환경 활성화 없이:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```
