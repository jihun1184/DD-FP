# FROZEN_DEPENDENCIES — 이 패키지가 동결 입력에 의존하는 부분

이 패키지는 새 실험을 하지 않는다. 아래 항목은 **이전 실행의 기록 또는 동결 소스**이며, R2.3에서 새로 만든 것이 아니다.

| 입력 | 쓰이는 곳 | 고정 방법 | R2.3에서 한 일 |
|---|---|---|---|
| `gpu_native_worklist.py` (SHA-256 853bf33a…17163ac), `cpu_fp.py` (8eae554f…), `g8_gpufp_semantic_probe.py` (566655ce…), `g8_pit_example.json` (e327fcea…) | `evidence_phase1/*`(Fig. 2, 3, 5의 CPU 모델·열거·tie 사례) | `evidence_phase1/frozen_inputs/`에 복사본 + `frozen_check.py`가 해시 불일치 시 실패 | 해시 확인; `instr_pit.py`·`enum_critical.py`·`tie_cases.py`·`seam_example.py 3` 재실행(저장 JSON과 동일, `verify_r22_numbers.py`의 RERUN 4) |
| `exp_b1_corrected.csv` (SHA b64e4afafac84fe9…), `exp_b1_drive_threshold_grid.csv` | Table 1, S10, CREMI-2D 문장 | `evidence_drive/` | 이미지별 기록에서 집계 재계산(Part-B 파이프라인 재실행 아님) |
| `FIG4_DWC_MECHANISM_SUMMARY_FROZEN.json` | Fig. 6 | `evidence_drive/` | 기준 적용·순위 재계산 |
| `trace_out.json` | Fig. 4(b), S4 | `evidence_ibi/`; production 트리 재구성은 `REBUILD_PRODUCTION_TREE.md`(4개 파일의 SHA 고정) | **재실행하지 않음**; 읽고 단언만 함 |
| 보존된 R2.2 기본 트리 | 청구 손실 비교(`claim_loss_check.py`) | 인자 `BASE` | 변경하지 않음 |

**빌드.** `bash tools/build_r23.sh <BASE_R2_2> <OUT>`는 상대 경로와 `mktemp`만 쓰며 세션 임시 경로에 의존하지 않는다. 필요한 TeX 패키지: elsarticle, tikz(patterns·arrows.meta), cleveref, algorithmicx 등 원고 서문 참조. 절대 경로는 검사 출력에서 `<TREE>`/`<BASE>`로 가린다.

**하지 않는 말.** 동결 소스가 올바르다는 것, trace가 현재 코드에서 재현된다는 것, CUDA 실행이 CPU 모델과 같다는 것은 이 문서가 보증하지 않는다.
