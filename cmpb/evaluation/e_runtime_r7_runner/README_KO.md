# DDFP_R7_GPU_SERVER_RUNNER_R0 — GPU-native FP 검증 서버 실행 패키지 (R7)

**상태(먼저 읽을 것).** 이 패키지는 GPU가 없는 환경에서 만들었습니다. 제작 환경에서 수행한 것은 정적 검사와 CPU 전용 시험(`bash run7.sh selftest`)뿐이며 (scipy 가 없는 파이썬에서는 production 모듈을 import 하는 stub 시험 12개가 SKIP 으로 표시되고 PASS 로 세지 않습니다)
**CUDA 실행은 단 한 번도 수행되지 않았습니다 (NOT RUN)**. E1/E2/E6/E3/E4의 GPU 결과는 서버에서 실행하기 전까지 모두 **OPEN**입니다.
CPU 셀프테스트의 "PASS"는 패키지(판정 로직·음성 대조군·게이트)에 대한 것이며 CUDA 정확성의 증거가 아닙니다.
기존 동결 코드·원고는 수정하지 않았습니다. R3 파일은 바이트 그대로 재사용했고(`MANIFEST_R3_REUSED.txt`로 R3 해시와의 일치를 확인), R7 신규 파일은 `r7/`, `run7.sh`, `config/r7.env.example`, `README_KO.md`뿐입니다.

## 1. 판정 구조 (알고리즘은 바꾸지 않음)

| 단계 | 무엇을 하는가 | 게이트 |
|---|---|---|
| preflight | 읽기 전용 점검: 패키지 해시, 파이썬/CuPy, GPU 점유(타 작업 유무), RAM/디스크, 데이터·코호트 SHA-256, **서버의 기존 DD-FP 트리가 봉인 기준선과 같은지** | — |
| prepare | 번들 + 동결 PATCH로 봉인 체크아웃 `DD-FP-R7`을 오프라인 구성, 6개 SHA-256 확인 | — |
| e2 | **실제 CUDA** GPU parity 50 fixture × tie 2종 = 100 run (3×3×5 반례 포함). 비교: 출력 비트, tie 이벤트/level transition, seed, 상태, DWC, e2e | preflight PASS |
| smoke | E1/E6/E4 스크립트의 CUDA 배선을 초소형 합성 볼륨으로 몇 초 점검. **증거 아님·게이트 미개방** | preflight PASS (e2 이후 권장) |
| e1 | BraTS N=20: GPU K=1 전체 배열 vs **같은 tie 정책의 CPU 상태기계**(직접 패리티), 동결 Sequential FP(의미 분석), 전역 DWC(GPU 검사기 + 독립 numpy 검사기) | e2 PASS |
| e6 | 같은 서버·같은 GPU에서 K=1 / DD-FP(K=16, δ=1) / Sequential FP, **3회 반복**, 워밍업·실행순서 무작위·메모리 계측, E3용 검증 비용 동시 기록 | e1 PASS |
| e3 | e6 행에서 파생: seam 스캔 vs 전역 검사 비용 (GPU 불필요) | e6 PASS |
| e4 | 안전한 단계적 메모리 확장. **의도적 OOM 없음.** `--confirm E4` 필요 | e6 PASS |

**E1은 세 축을 독립적으로 판정합니다.** (1) 구현 패리티: CUDA-native == 같은 tie 정책 CPU 상태기계(비트, tie 이벤트 수, level transition, backward 수). (2) DWC: 모든 출력에서 위반 0 이고 GPU 검사기 = 독립 검사기. (3) Sequential FP와의 동등성: **기술(descriptive)만 하며 게이트가 아닙니다.** Sequential과 CUDA-native는 tie 정책이 다르므로 출력 불일치를 CUDA 오류로 판정하지 않습니다. 게이트 = (1) AND (2).
불일치 시 분류(가설이며 증명이 아님): `SEQ_EQUALS_LOWER` / `SEQ_EQUALS_HIGHER_ONLY` / `TIE_SEMANTICS_CANDIDATE_NOT_PROVEN` / `DIFFERS_BUT_LOWER_EQUALS_HIGHER`(순수 lower·higher가 같은 출력인데 Sequential만 다름 → tie 선택만으로는 설명되지 않음, 조사 필요) / `UNEXPLAINED_NO_TIES`.
문헌 근거(RECORDED: 이 세션에서 웹 도구가 PDF 본문을 요약해 확인한 내용이며, 인용 전 원문 대조 필요): Boutry–Najman–Géraud, "Topological Properties of the First Non-Local Digitally Well-Composed Interpolation on n-D Cubical Grids"(LRDE 게재 원고 PDF, <https://www.lre.epita.fr/dload/papers/boutry.20.jmiv.1.pdf>; 학술지 표기는 "JMIV manuscript"로만 나오며 최종 게재 권호·쪽수는 **미확인**) §3.6 은 다음 레벨을 현재보다 작은 쪽/큰 쪽 어느 쪽이든 고를 수 있고 결과 영상은 달라지지 않는다고 서술하며, 등거리 tie 규칙은 명시하지 않고 증명은 §8 에 스케치만 있습니다. 즉 "tie 선택이 결과를 바꾸지 않는다"는 것은 **논문 주장(증명은 스케치)**이지 이 패키지에서 확인된 사실이 아닙니다.
셀프테스트(소형 볼륨 60개)에서 tie 이벤트는 흔했지만 lower/higher **출력**이 달라진 사례는 0건이었습니다(해당 입력에 한정된 관찰). 따라서 K=1에서 "Sequential ≠ CUDA"가 나오면 tie 설명은 자동으로 성립하지 않으며, 위 분류와 level-transition 비교로 확인합니다.

## 2. 서버에서 실행할 명령 (복사·붙여넣기 순서)

`sudo`는 필요 없습니다. 경로는 예시이며 원하는 위치로 바꾸십시오. 압축은 기존 데이터·결과·학습 폴더 밖에서 풉니다.

```bash
mkdir -p ~/ddfp_r7 && cd ~/ddfp_r7
unzip /path/to/DDFP_R7_GPU_SERVER_RUNNER_R0.zip && cd DDFP_R7_GPU_SERVER_RUNNER_R0
sha256sum ../*.zip 2>/dev/null          # 전달받은 SHA-256과 비교
bash run7.sh verify-pkg                 # "ok": true
bash run7.sh selftest                   # CPU 전용 약 1분. 마지막 줄 "SELFTEST PASS"

cp config/r7.env.example config/r7.env
nano config/r7.env    # 최소: R3_BRATS_ROOT=<BraTS2021 경로>  R3_OUT_ROOT=<결과 경로>  (R3_PROTECT_DIRS 에 기존 결과/학습 폴더 권장)
                      # R3_PYTHON=<numpy, scipy, nibabel, cupy 가 있는 venv/conda 의 python>  (새 환경이 필요하면: bash run7.sh setup-venv, 네트워크 필요)
                      # 다른 GPU 작업이 있으면 R3_CUDA_DEVICE=<nvidia-smi 번호> (auto = 유휴 GPU 자동 선택, 점유된 GPU 는 절대 사용하지 않음)

# R7-B  읽기 전용 서버 점검 (CUDA 커널·메모리 할당 없음)
bash run7.sh preflight --server-tree <기존 서버의 DD-FP 작업 트리>    # 트리를 비교하지 않고 봉인 체크아웃만 쓰려면 --no-server-tree
```

`PREFLIGHT PASS`가 아니면 `[NG]` 항목을 해결하고 같은 명령을 다시 실행합니다(새 attempt 폴더가 생기고 이전 것은 보존됩니다).
`server_tree.equals_sealed_baseline`이 NG이면(서버 코드가 봉인 기준선과 다름) **어떤 실험도 시작되지 않으며**, 그 상태를 `server_paths.json`과 함께 보내 주십시오. 실제 서버 경로(패키지, 설정, 데이터, 결과, 파이썬, 서버 트리, GPU)는 `…/preflight/attempt_*/server_paths.json`에 기록됩니다.

```bash
bash run7.sh prepare                    # 봉인 체크아웃 DD-FP-R7 (오프라인). 이미 있으면 검증만 하고 아무것도 바꾸지 않음
# R7-C  단계별 실행 — 앞 단계가 PASS 일 때만 다음 단계가 열립니다
bash run7.sh e2                         # 약 10 분 (미측정 추정)
bash run7.sh smoke                      # 몇 초~몇 분. 선택이지만 권장 (E1 에서 몇 시간 뒤 배선 오류가 나는 것을 방지)
bash run7.sh e1                         # 약 2–2.5 시간 (미측정 추정, 아래 표)
bash run7.sh e6                         # 약 2.5–3 시간 (미측정 추정). 실행 중 서버 CPU 에 다른 무거운 작업이 없어야 함
bash run7.sh e3                         # 즉시 (e6 데이터에서 파생)
bash run7.sh e4 --confirm E4            # 별도 자원 점검 후에만. 약 0.5–1 시간
bash run7.sh status                     # 단계별 판정 요약
bash run7.sh collect                    # 결과 ZIP + SHA-256 (150 MiB 초과 파일은 서버에 남기고 목록만 기록)
bash run7.sh verify-results --results <수집 ZIP 또는 폴더>   # 원자료에서 판정을 재계산 (서버 밖에서도 가능)
```

긴 단계는 `tmux`/`screen` 안에서 실행하고 `R7_NO_PAUSE=1`을 지정하면 실패 시 Enter 대기를 하지 않습니다. 대화형 터미널에서는 비정상 종료 시 창이 닫히지 않고 Enter를 기다립니다. 모든 콘솔 출력은 결과 폴더의 `console_*.log`에도 저장됩니다.
`e1`/`e6`은 중단(Ctrl-C, 정전, timeout 등) 후 `--resume`으로 이어서 실행합니다: 끝나지 않았거나 `INCOMPLETE`인 마지막 attempt를 같은 폴더에서 계속하며(완료된 피험자/행은 보존, 이전 `verdict.json`·봉인 파일은 `*.before_resume_*`로 이름만 바뀌어 남음), PASS/FAIL/INVALID attempt는 `--resume`으로 다시 열리지 않습니다.
E6 반복을 늘리려면 PASS 이후 `bash run7.sh e6 --extend --reps-start 3 --reps 2` 로 같은 attempt에 4–5회째를 추가합니다(판정은 모든 방법·피험자에 공통인 반복만 사용하고 다시 봉인됩니다). e3/e4 는 그 뒤에 새로 실행하십시오.

### 예상 자원 (모두 **미측정 추정**)

시간은 서버 CPU의 Sequential FP 속도에 지배됩니다. 기록된 벤치마크 평균은 Sequential 111.5 s/피험자(CPU 모델 기록 없음)이지만, 패키지 제작 환경(Xeon 2.8 GHz급 클라우드 CPU)에서 동결 `fp_cpu`(순수 Python)를 40³ 랜덤 볼륨으로 재 보면 확장 셀당 약 4–6.5 µs(OBSERVED, 이 환경 한정)였고, 이를 기록 평균 71 M 셀에 외삽하면 한 번에 300–470 s(외삽, 실제 BraTS 볼륨은 평탄 배경이 많아 다를 수 있음)입니다. 두 값이 3–4배 다르므로 **`preflight`가 서버 CPU에서 같은 프로브를 돌려 `cpu.sequential_speed_probe`로 E1/E6 CPU 시간을 미리 출력**합니다. 그 값을 보고 시작 여부를 정하십시오.

| 단계 | 시간 | VRAM | 디스크 / RAM | 비고 |
|---|---|---|---|---|
| preflight/prepare | <10 분 | — | <1 GB | |
| e2 | ~10 분 | 소규모 | 무시 | NVRTC 컴파일 포함 |
| smoke | 수 초–수 분 | 소규모 | 무시 | 증거 아님 |
| e1 | CPU 부분 = 피험자당 3회 순차 실행(동결 Sequential + 같은-tie 상태기계 lower/higher) × 20 = 60회. 기록 속도 기준 ≈ 2–2.5 h, 제작 환경 속도 기준 ≈ 6–8 h. GPU 부분 ≈ 10 분 | 기존 N=20 이력상 24 GB 내 (peak 이번에 측정 안 함) | 해시·JSON <100 MB. **불일치 시에만** 배열 저장(상한 10 GiB) | 단일 스레드 Python. `--resume`으로 분할 실행 가능 |
| e6 | Sequential 60회 (기록 속도 ≈ 1.9 h, 제작 환경 속도 ≈ 6–8 h) + GPU 약 40–60 분(검증 비용·메모리 패스 포함) | 방법별 peak 를 이번에 측정 | 로그 <100 MB | **E6 동안 서버 CPU 에 다른 무거운 작업이 없어야 함** |
| e4 | 0.5–1 h | 사용 가능 VRAM 의 75 %·잔여 3 GiB 이내로만 확장 | 무시 | 의도적 OOM 없음 |

`R3_MIN_DISK_GIB=15`(예시 설정에 적용)는 불일치 시 저장될 수 있는 배열을 고려한 값입니다.

## 3. 판정값과 STOP 조건 (사전 등록)

판정: `PASS` / `FAIL` / `INVALID`(GPU 증거가 아니거나 무결성 문제) / `INCOMPLETE`. 다음 단계는 **봉인된 PASS**(같은 패키지 해시, 같은 GPU uuid, 봉인 후 `verdict.json` 미변경)에서만 열립니다.

| 단계 | PASS 조건 | STOP / 해석 |
|---|---|---|
| e2 | backend=cupy 이고 GPU 이름 존재, 100/100 run, 모든 비교 범주 동일, vendor·fixture·script 해시 일치, CPU 결정적 | 불일치 → **FAIL = CUDA 정확성 문제를 먼저 해결**, e1 이하 닫힘. CPU 에뮬레이션/stub 은 **INVALID** |
| e1 | 패리티(두 tie 정책 모두) AND DWC(모든 출력, GPU 검사기 = 독립 검사기) AND 무결성(입력 해시, 6개 생산 해시, ispan GPU=CPU, 외부 GPU 프로세스 없음) | 패리티 FAIL → CUDA 구현/상태기계 불일치 조사. DWC FAIL → 위반 위치 조사. **Sequential 불일치는 FAIL 사유가 아님**(분류만 보고) |
| e6 | 측정 유효성: 20명 × 공통 반복 ≥3 × 3방법, 출력 해시가 반복 간 동일, **E1 해시와 동일**, 외부 GPU 프로세스 없음 | 속도 문장은 측정 사실로 부트스트랩 CI(B=10000, seed 20261014)와 함께 보고. `more_repetitions_recommended`가 true 면 반복 추가 |
| e3 | 데이터 유효성만 | seam 스캔(CPU numpy, 절단면 xz/yz face 블록만)과 전역 검사(GPU, 전 방향 face+cube)는 **같은 성질을 인증하지 않음** — 모든 문장에 이 단서가 붙음 |
| e4 | 예상치 못한 OOM/ERROR 없음, 방법별 ≥1 단계 완료, 출력 유한·shape 정확, 외부 GPU 프로세스 없음 | 안전 규칙으로 중단된 단계(`SKIPPED_BY_SAFETY_RULE`)는 **처리 한계의 하한**일 뿐 실제 한계가 아님 |

실행 중 다른 사용자의 GPU 프로세스가 나타나면 러너는 **자기 자식 프로세스만** 멈추고(타 프로세스는 건드리지 않음) 종료 코드 4로 끝납니다. 시작 시 GPU 에 다른 작업이 있으면 어떤 GPU 단계도 시작하지 않으며 attempt 폴더도 만들지 않습니다.

실험 단계가 FAIL 인 뒤에는 다음 단계가 닫힙니다. 원인을 판단한 뒤에는 (a) 원인 수정 후 같은 단계 재실행, 또는 (b) `--triage-note "<판단 내용>"`으로 다음 단계를 엽니다(기록됨, e2/e1/e6 의 FAIL 에만 허용; preflight FAIL·INVALID·INCOMPLETE 에는 불가). 실패한 attempt 폴더 전체를 보내 판단을 받은 뒤 진행하는 것을 권장합니다.

## 4. 측정 방법 (E6/E4)

- 시간: 단계별 동기화(`cp stream sync`)로 `t_prep / t_fp / t_d2h / t_ibi / t_out`을 분리. 전처리와 검증 비용(seam, H2D, 전역 검사)은 알고리즘 시간에 포함하지 않고 별도 기록 → 데이터 준비·검증 시간과 알고리즘 시간이 분리됨.
- 메모리: 시간 측정 구간은 무간섭(`/proc/self/clear_refs` 로 VmHWM 재설정, CuPy 풀 바이트). 폴링 샘플러(`cudaMemGetInfo`)는 비시간 메모리 패스와 E4 에서만 사용.
- 실행 순서: 피험자·반복별 무작위(시드 20260904), 방법별 워밍업 1회(비측정). Sequential 기준선은 **동결 `fp_cpu` 단일 스레드**이며 최적화된 CPU 구현이 아닙니다(이 단계에서 최적화 CPU 기준선은 실행하지 않음).
- E4: `예측 = 1.15 × max(비례, 선형 외삽, 멱법칙 외삽)`이 VRAM 한도 `min(0.75·총량−기저, 총량−기저−3 GiB)`, RAM 한도 `0.6·가용`, 시간 예산을 넘으면 그 단계를 실행하지 않습니다. 이 예측은 안전 여유이지 보장이 아닙니다(예측을 넘어서 가속되는 성장은 한도 여유분이 흡수).

## 5. 이 패키지가 증명하지 못하는 것

- K=16 슬랩 루프는 직렬이므로 K=16 병렬 실행·속도 향상을 주장하지 않습니다. DD-FP 출력은 Sequential FP/K=1 출력과 같지 않습니다(E1 이 다시 기술).
- 전체 블록 단위 인증(타일 내부 + cross-owner 블록 전부)은 production 에 구현되어 있지 않으며 E3 는 구현된 seam 검사만 비교합니다.
- DWC 는 원본의 Betti 수/호모토피 보존을 뜻하지 않습니다. DWC 인증·Sequential 출력 일치·위상 보존은 서로 다른 평가 목표입니다.
- E2/E1 의 PASS 는 50×2 fixture 및 BraTS 20명에 대한 유한 증거이며 일반 동치 증명이 아닙니다.
