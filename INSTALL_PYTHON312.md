# Python 3.12 설치 가이드

`chromadb`를 사용하기 위해 Python 3.12가 필요합니다.

## 설치 방법

### 방법 1: 공식 웹사이트에서 다운로드 (권장)

1. 다음 링크에서 Python 3.12를 다운로드하세요:
   - https://www.python.org/downloads/latest/python3.12/
   - 또는 직접: https://www.python.org/ftp/python/3.12.12/python-3.12.12-amd64.exe

2. 다운로드한 설치 파일을 실행하세요.

3. **중요**: 설치 시 다음 옵션을 반드시 체크하세요:
   - ✅ **"Add Python 3.12 to PATH"** (PATH에 Python 3.12 추가)
   - ✅ **"Install for all users"** (선택사항, 관리자 권한 필요)

4. 설치가 완료되면 터미널을 다시 열고 다음 명령어로 확인하세요:
   ```powershell
   py -3.12 --version
   ```
   출력: `Python 3.12.x`

### 방법 2: py launcher를 사용한 자동 설치 (Windows 10/11)

Windows의 `py` launcher를 사용하여 Python 3.12를 설치할 수 있습니다:

```powershell
# Python 3.12 설치 (Microsoft Store에서 자동 다운로드)
py install 3.12
```

또는:

```powershell
# winget을 사용한 설치 (Windows 10/11)
winget install Python.Python.3.12
```

## 설치 확인

설치 후 다음 명령어로 확인하세요:

```powershell
py -3.12 --version
python --version  # 기본 Python 버전 (3.14.2일 수 있음)
py -0  # 설치된 모든 Python 버전 목록
```

## 가상환경 재생성

Python 3.12가 설치되면 다음 단계를 수행하세요:

1. 기존 가상환경 삭제:
   ```powershell
   Remove-Item -Recurse -Force venv
   ```

2. Python 3.12로 새 가상환경 생성:
   ```powershell
   py -3.12 -m venv venv
   ```

3. 가상환경 활성화:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. 패키지 설치:
   ```powershell
   pip install -r requirements.txt
   ```

또는 `run.bat`를 실행하면 자동으로 처리됩니다:
```powershell
.\run.bat
```

## 문제 해결

### "python -m venv venv" 실행 시 `can't open file '...WindowsApps\python.exe' [Errno 22]` 오류

**원인**: PATH에서 `python`이 Windows 앱 별칭(WindowsApps\python.exe)으로 먼저 잡혀, venv가 그 경로를 사용하려다 실패합니다. 실제로 실행 중인 인터프리터는 `pythoncore-3.12-64`입니다.

**해결**: 인터프리터를 명시해서 venv를 만드세요.

```powershell
# 방법 1: py 런처 사용 (권장)
py -3.12 -m venv venv
```

```powershell
# 방법 2: 실제 Python 실행 파일 경로 사용
& "C:\Users\asakh\AppData\Local\Python\pythoncore-3.12-64\python.exe" -m venv venv
```

(다른 사용자/설치 경로면 해당하는 `python.exe` 경로로 바꾸세요.)

### `[Errno 13] Permission denied: '...\\venv\\Scripts\\python.exe'` 또는 venv 삭제 시 "Access denied"

**원인**: 다른 프로세스가 `venv\Scripts\python.exe`를 사용 중이라 파일이 잠겨 있습니다. (예: Cursor/VS Code의 Python 인터프리터, 백신, 이전에 켜 둔 터미널.)

**해결** (아래 순서대로 시도):

1. **Cursor/VS Code 완전 종료**  
   IDE를 닫은 뒤, **Cursor를 쓰지 않는** 새 PowerShell 또는 명령 프롬프트를 연다.

2. **프로젝트 폴더로 이동 후 venv 삭제**  
   ```powershell
   cd C:\Users\asakh\Documents\GitHub\AI_agent_lecture_final
   Remove-Item -Recurse -Force venv
   ```

3. **다시 venv 생성**  
   ```powershell
   py -3.12 -m venv venv
   ```

4. **Cursor를 다시 연 뒤** 가상환경 활성화:  
   `.\venv\Scripts\Activate.ps1`  
   그리고 필요하면 Cursor에서 인터프리터를 `venv\Scripts\python.exe`로 다시 선택.

IDE를 닫지 않으려면: Cursor에서 **Python 인터프리터를 시스템 Python**(예: `py -3.12` 또는 `pythoncore-3.12-64\python.exe`)으로 바꾼 뒤, 터미널에서 위 2–3단계를 실행해 보세요. 그래도 삭제가 안 되면 1번처럼 Cursor를 완전히 종료한 뒤 삭제·재생성하면 됩니다.

### "py -3.12"가 작동하지 않는 경우

1. Python 3.12가 올바르게 설치되었는지 확인:
   ```powershell
   py -0
   ```

2. Python 3.12의 전체 경로를 사용:
   ```powershell
   # 일반적인 설치 경로
   C:\Users\<사용자명>\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
   ```

3. 또는 환경 변수 PATH에 Python 3.12가 추가되었는지 확인
