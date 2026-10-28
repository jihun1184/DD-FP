# GPU parity (P1.1 gate, item C) — CPU reference vs GPU, bit for bit

**범위:** 3×3×5 반례 + 고정 fixture 49개 = 총 50개 (`fixtures.json`, 내부 SHA-256 포함), 각각 tie policy `lower`/`higher` → 100 run.
소규모 정확성 검사이며 **벤치마크가 아닙니다.** N=20·Full-N 실험은 포함하지 않습니다. 관리자 권한/sudo 불필요.

## 필요한 것
- Windows + NVIDIA 드라이버(RTX 3080 Ti Laptop), Python 3.10–3.14 (CuPy 공식 지원 범위: https://docs.cupy.dev/en/stable/install.html)
- 패키지: `numpy`, `cupy-cuda12x[ctk]` (`[ctk]`는 CUDA 구성요소를 pip로 설치해 시스템 CUDA Toolkit 없이 NVRTC 커널 컴파일을 가능하게 합니다. 공식 설치 페이지 기준. 라이브러리 구성은 해당 페이지에 명시되어 있지 않으므로 실패 시 아래 "문제 해결" 참조)
- CuPy 패키지는 **하나만** 설치 (`cupy`와 `cupy-cudaXX` 동시 설치 금지)

## 실행 (PowerShell 또는 cmd, 이 폴더에서)
```
# 0) 1회 설치 (venv, 관리자 권한 불필요)
py -3 -m venv .venv_parity
.venv_parity\Scripts\python.exe -m pip install -U pip setuptools
.venv_parity\Scripts\python.exe -m pip install numpy "cupy-cuda12x[ctk]"

# 1) CPU 전용 자체 점검 (GPU 불필요; 무결성 해시 + CPU 결정성 + CPU 다이제스트)
.venv_parity\Scripts\python.exe -I run_gpu_parity.py --selfcheck --no-pause

# 2) GPU parity 본 실행
.venv_parity\Scripts\python.exe -I run_gpu_parity.py --no-pause
```
같은 일을 하는 래퍼: `run_gpu_parity.bat setup` → `run_gpu_parity.bat selfcheck` → `run_gpu_parity.bat run` (끝나면 `cmd /k`로 창이 유지됩니다).
일부만 실행: `--only F00` (접두사 일치). 결과: `results_gpu\<시각>_cupy\parity_report.json`.

## 판정 기준
- 종료 코드 `0` 전체 일치 / `1` 불일치 또는 예외 / `2` 환경·무결성 문제(아무것도 실행 안 함) / `3` 내부 오류. 어떤 경우에도 창은 닫히지 않습니다(예외는 fixture 단위로 잡아 기록하고 계속 진행).
- 비교 항목(모두 CPU 참조 `vendor/cpu/seedref.py` 대비): **out**(float32 비트 단위 동일) · **tie**(tie 이벤트 수, level transition, direction/activation count) · **seed**(seed 수, 활성화 수, raw-bit 보존, `is_seed`, scheduled/injection level) · **state**(discovered/processed/pending) · **dwc**(출력의 face/cube 위반 수) · **e2e**(IBI free-running 전체 조립 결과·라운드 수).
- 두 모드: *e2e*(GPU가 자기 출력을 다음 라운드 pin으로 사용) + *per-solve*(CPU가 기록한 입력을 GPU가 재풀이 → 오차 전파 분리).
- F00은 저장된 반례(`mismatch` 18셀, DD-FP=52 vs FP=57)를 CPU·GPU 모두 재현하는지도 확인합니다.
- F41–F49는 CPU 참조에서 equidistant tie 이벤트가 실제로 발생하는 tile입니다(무작위 3×3×D tile의 약 1.4 %, 알파벳 8). 무작위 체적 21,357 tile 중 tie 이벤트 3건·`lower`/`higher` 출력 차이 0건이어서, 최종 출력이 tie policy에 의존하는 fixture는 만들지 못했습니다.
- `cpu_digest`: 모든 CPU 출력의 해시. `expected_cpu_digest.txt`(Linux/numpy 2.2.6 VM에서 생성)와 다르면 numpy/플랫폼 차이이며, GPU-vs-CPU 판정 자체는 유효합니다.

## 무결성 / 출처
- `vendor/VENDOR_SHA256.txt`: 실행 시 모든 vendor 파일을 검증(불일치 시 중단). `cpu/*`는 `E_repro/src`의 바이트 복사본(동결 해시 원장과 일치), `gpu/*`는 P3 패치에서 재구성한 `gpu_native_worklist.py`(853bf33a…), `gpu_seed_aware_worklist.py`(c80f56b9…).
- 이 harness는 **VM에서 CPU stub으로만 검증**했습니다(stub: 100/100 통과, 의도적 결함 주입 stub: 18/100 실패로 검출). **실제 GPU에서는 아직 실행하지 않았습니다.**

## 문제 해결
- `ModuleNotFoundError: cupy` → 0) 설치 수행. `nvrtc`/헤더 오류 → `cupy-cuda12x[ctk]`로 재설치하거나 CuPy 문서의 `nvidia-cuda-runtime-cu12` 안내 참조.
- 실패 시 창의 전체 출력과 `parity_report.json`을 그대로 전달해 주세요.
