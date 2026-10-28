# G5 사전등록 (v2) — 동결 K=1 · AF · FP-aware generic pile

작성: 2026-10-23 (G5 GPU 수치가 하나도 없는 시점). 이 문서와 `g5_rules.py`의 상수는 결과를 보기 전에 고정한다. 실행 후 변경은 "수정 기록(Amendment)"으로만 허용하며 원래 규칙의 판정을 대체하지 않는다.
`comparison/run_generic_gpu.py`(v1)의 규칙은 GPU에서 한 번도 실행되지 않았고 본 문서로 대체된다. v1 코드는 삭제·수정하지 않는다.

## 1. 질문

Q-G5. 동결 FP 계약(최근접 레벨 선택·first-discovery clamp·라운드 경계·tie=lower·상태 배열)을 **알고 있는** 일반 pile-and-split worklist가 (a) 동결 K=1과 같은 출력·보조 배열·카운터·레벨 전이열·closure 라운드를 실데이터 규모에서 보존하는가, (b) AF보다 느린가/비슷한가/빠른가, (c) 메모리는 얼마나 쓰는가.

범위 한정: AF는 **K=1 unseeded 경로**에만 적용된다. 이 실험은 K>1 DD-FP 전체 가속에 대해 아무것도 말하지 않는다(R7 감사: 분해 K=16은 K=1보다 빠르지 않았음).

## 2. 비교 대상과 호출 (타이밍 범위 동일)

| 방법 | 호출 | 입력 | 반환 |
|---|---|---|---|
| frozen | `boutry_bucket_gpu(cp, lo, hi, l, "lower", 256)` | G3와 동일한 `env.prep(vol)` 결과 | (out, meta, aux) 호스트 배열 |
| af | `boutry_active_frontier(cp, lo, hi, l, "lower", 256, cap=4,194,304, frozen=F)` | 동일 | 동일 |
| pile | `generic_pile_worklist(cp, lo, hi, l, "nearest-lower", 256, list_cap=4,194,304, pile_cap=N)` | 동일 | 동일 |

세 방법 모두 내부에서 같은 `_validate_host_intervals`(lo/hi D2H)를 거치고, 끝에서 out과 aux 5개를 D2H한다. 타이머는 호출 직전부터 `env.sync()` 직후까지이며, 동결 대비 비교(검증)는 **타이머 밖**에서 한다. G3의 동결 측정은 `be._tie(...)`를 통했으나 G5는 세 방법 모두 직접 진입점을 쓴다(의도된 변경, G3 대비 일치는 `g3_consistency`로 점검).

## 3. 상태 분류 (방법별·실행별로 정확히 하나)

`OK` · `CORRECTNESS_FAIL`(동결 참조와 출력/보조/카운터/전이열/closure 불일치) · `RESOURCE_FAIL`(용량 초과, OOM) · `IMPL_ERROR`(그 밖의 예외: 불변식 위반, 컴파일 오류 등) · `CUDA_FAULT`(CUDA 런타임/드라이버 오류, 불법 메모리 접근; 즉시 중단).
경고: 실패는 **구현·구성의 사실**이다. 어떤 실패도 "일반 방식의 필연적 한계"나 "AF 설계의 필요성"으로 해석하지 않는다. v1의 `AF-NECESSARY` 라벨은 삭제했다.

## 4. 예열·재현성

- 별도 예열(24³ 중앙 크롭): CUDA 컨텍스트 → 방법별 cold 호출 → warm 호출. `init_cost_s = cold − warm`을 기록하고 timed 행에 넣지 않는다.
- 피험자마다 timed 이전에 전체 크기 무시간 확인 실행(frozen 참조 → AF → pile).
- 반복 수는 3의 배수(기본 6). 실행 순서: 3회 = 무작위 순열의 3개 회전, 6회 = 6개 순열 전부. 모든 방법이 모든 위치에 같은 횟수 등장.
- 각 timed 실행 전 `env.free(); env.sync()`(G3와 동일).
- 모든 실행(예열 제외 시간만)은 동결 참조와 `compare()`로 전수 대조: 출력 바이트(SHA-256), `META_KEYS` 전부, aux 5개 배열, aux 키 집합.
- 행 완결성: (피험자 × rep × 방법)당 timed 행 정확히 1개, 초과·누락은 `INVALID-RUN`.

## 5. 단계

- **Stage 1**: G3 첫 피험자(BraTS2021_00147) 1건. `STAGE1_PASS.json`은 (i) CUDA 실행, (ii) 3방법 전 실행 `OK`, (iii) 행 완결, (iv) 예열 통과, (v) 방법별 VRAM 측정 존재일 때만 기록된다.
- **Stage 2**: G3와 같은 20건, 같은 순서. `STAGE1_PASS.json`의 코드 SHA-256 전부와 GPU 이름이 일치해야 시작한다.
- Stage 1에서 pile이 실패하면 확대하지 않는다. 먼저 원인을 진단하고(§7), 코드를 고치면 Stage 1부터 다시 시작한다.

## 6. 측정 항목

시간: 방법·피험자·rep별 `t_total_s`; 피험자별 중앙값; `r = t_AF / t_PILE`(피험자 중앙값 기준); 동결 대비 가속.
메모리(피험자별 1회, 방법별): 장치 전체 VRAM 최대 증가량(cudaMemGetInfo 2 ms 폴링, 프로젝트 샘플러 병행), CuPy 풀 통계. pile은 추가로 배열별 할당 바이트와 실제 점유(최대 pile, 최대 frontier, 최대 다음 리스트 대 cap).
분해(intrusive, 헤드라인 아님): 커널별 GPU 시간(이벤트), 실행 횟수, 실행 스레드 수(메모리 트래픽 대리변수), 호스트 sync 횟수·시간, D2H 횟수·시간; AF는 자체 profile(gather/round/finalize)도 병행. 이 패스의 실패는 판정에 영향을 주지 않는다.
G3 일관성: 이번 실행의 AF/frozen 중앙값 가속 대 G3(4.632)의 상대차, 10% 초과 시 경고(판정 불변).

## 7. 판정 규칙 (g5_rules.py V2.0, 고정값)

순서대로 평가한다.

1. 비CUDA 실행 → `NOT-MEASURED` (shim은 절대 판정이 아님).
2. `CUDA_FAULT` 존재 → `ABORTED-CUDA-FAULT` (환경 대 pile 커널 진단 필요, 연구 추론 없음).
3. frozen 실패 → `INVALID-REFERENCE-FAILURE`.
4. AF 실패(불일치 포함) → `INVALID-AF-REGRESSION` (G3/G4와 모순; 어떤 G5 추론도 하지 않음).
5. pile에 `CORRECTNESS_FAIL` 또는 `IMPL_ERROR` → `PILE-IMPLEMENTATION-FAILURE`. 필수 진단 4항목: 커널 결함(경합/원자연산/인덱싱) · 용량/구성 · FP 계약 명세 오류 · 하네스/타이밍 범위 결함. 속도 판정 없음.
6. pile에 `RESOURCE_FAIL` → `PILE-RESOURCE-LIMITED` (구성 한정, 실패 피험자의 의미는 미검증, 속도 판정 없음, 기술 통계만).
7. timed 행 불완전 → `INVALID-RUN`.
8. 피험자 20 미만, reps < 3, 또는 G3 피험자 집합·순서 불일치 → `NOT-DECIDABLE`(정확성·자원 결과만 유효).
9. 소음: 셀(피험자×방법)의 (최대−최소)/중앙값 > 10%인 셀이 전체의 20% 초과 → `INDETERMINATE-NOISY`.
10. 속도(피험자 20명 모두 pile OK): `r_i`의 중앙값에 대해 피험자 부트스트랩 95% CI(B=10,000, seed 20261023), 동등 한계 **1.10**.

| 조건 | 판정 |
|---|---|
| CI 하한 ≥ 1.10 | `PILE-FASTER` |
| CI 상한 ≤ 1/1.10 | `AF-FASTER` |
| CI 전체가 [1/1.10, 1.10] 안 | `EQUIVALENT` |
| 위 셋이 아니고 `r_i ≥ 1.10`인 피험자 ≥ 3 이고 `r_i ≤ 1/1.10`인 피험자 ≥ 3 | `MIXED-SUBJECT-DEPENDENT` |
| 그 밖 | `INDETERMINATE` |

경고(판정 불변): 실행 위치 효과 3% 초과, G3 가속 대비 10% 초과 차이.

## 8. 판정이 연구 결론에 미치는 영향 (사전 지정)

| G5 판정 | 연구 해석 | 하지 않는 것 |
|---|---|---|
| `PILE-FASTER` | 일반 FP-aware 활성 실행이 AF보다 빠르고 같은 의미를 보존 → 기여를 "계약 보존 활성 실행(설계 비의존)"으로 재평가. AF의 gather는 FP 특화 메커니즘으로 주장하지 않음. | pile형 커널을 production에 도입하거나 새 GPU 알고리즘을 설계하는 것(별도 승인) |
| `AF-FASTER` | 원인을 분해: 커널 GPU 시간 / 실행 스레드 수(메모리 트래픽) / sync·D2H 횟수와 시간 / 큐 처리(split·compaction 대 gather)로 나눠 보고. 한 성분이 시간 차의 50% 이상을 설명할 때만 그 성분을 원인으로 기술. | 구현 세부(예: 리스트 크기, per-round sync 수)를 알고리즘 필연성으로 해석 |
| `EQUIVALENT` | 속도 차별점 없음. AF 고유 가치는 메모리(부호와 크기)와 fallback 보험뿐. | 속도 우위 서술 |
| `MIXED-SUBJECT-DEPENDENT` | r과 피험자 특성(visits, closure 라운드, 최대 pile)의 순위상관을 기술 통계로만 보고. | 일반 결론 |
| `PILE-IMPLEMENTATION-FAILURE` | §7-5의 진단. 수정 시 Stage 1부터 재실행, 수정 이력 기록. | AF 필연성 주장 |
| `PILE-RESOURCE-LIMITED` | 메모리 보고와 구성 한정 서술. | 의미 실패로 해석 |

## 9. 실행 전 비구속 기대값 (사전 기록, 판정에 쓰지 않음)

- pile 상태·작업 배열 할당(분석값, N = 71,953,271): u, discovered, processed, scheduled_level 각 274.5 MiB(소계 1098 MiB) + pile 2개 549 MiB + 리스트 2개 32 MiB ≈ 1679 MiB. 동결(5개 N-배열)은 1372.5 MiB → **pile은 동결보다 약 +307 MiB** 더 쓸 것으로 예상(풀 반올림·입출력 제외). 관측 VRAM 차가 이와 크게 다르면 계측 또는 할당 모델 오류를 의미한다.
- 속도: G3 프로파일의 gather 비중(중앙값 39.6%, 26–47%)과 합성 프록시 pile 비율(N의 6–12%)을 바탕으로 한 모델 추정은 r ≈ 1.13–1.49. 이는 **측정이 아니며**, 실 BraTS의 pile 크기와 sync 지연이 모델을 벗어날 수 있다. 판정은 §7의 규칙만 따른다.
- AF peak 추가 메모리는 G3의 최대 32 MiB와 같은 수준.

## 10. E1과 파이프라인 측정 (G5 이후 판단 규칙)

- E1(FP 대 Min/Max 필요성)은 CPU 전용이고 G5 결과와 독립이다. "FP가 필요한 이유"를 증명하는 유일한 실험이므로 G5 결과와 무관하게 수행을 권고한다(승인 필요).
- 실제 파이프라인 측정(`run_pipeline_share.py`)은 GPU가 필요하다. Stage 2와 같은 GPU 세션에서 실행하되, 실제 다운스트림 단계를 `--extra-stage`로 지정할 수 있을 때만 의미가 있다. 판정 규칙 없음(기술 통계).

## 11. 수정 기록

(비어 있음 — 이 줄 아래에만 추가한다.)
