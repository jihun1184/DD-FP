# G5 v2 실행 절차 (RTX 4090 서버, 사용자 실행 — 작성자는 GPU에서 실행하지 않았음)

모든 명령은 `DDFP_A1_ACTIVE_FRONTIER` 폴더에서 실행한다. `...`는 G3 때 쓴 값과 동일하게 채운다(`--runner-dir`, `--repo`, `--lib`, `--brats-root`, `--metadata`).
기존 `frozen/`, `src/af_worklist.py`, production, R2.3은 수정하지 않는다. 출력 폴더는 매번 **새 폴더**여야 한다.

## 0. 사전 점검 (GPU 없이, 1–2분)
```
python3 -I comparison/test_g5_v2.py --out OUT_G5_TEST          # 53/53 PASS 이어야 함
sha256sum comparison/g5_rules.py comparison/generic_worklists_v2.py comparison/run_g5_v2.py src/af_worklist.py frozen/gpu_native_worklist.py
#  -> results/g5_v2/G5_V2_FILES.sha256 와 일치해야 함 (src/af_worklist.py 7b0a6529…, frozen 853bf33a… 는 기존 값)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # 다른 컴퓨트 프로세스가 없어야 함
```

## 1. Stage 1 — BraTS 1건 (G3 첫 피험자 BraTS2021_00147), 약 5분 예상(추정)
```
mkdir OUT_G5_STAGE1
python3 -I tests/collect_env.py --out OUT_G5_STAGE1/environment.json --files <G3 summary.json>
python3 -I comparison/run_g5_v2.py --backend cupy --stage 1 \
   --runner-dir ... --repo ... --lib ... --brats-root ... --metadata ... \
   --g3-summary <gate_4090_n20_20261022_180342/summary.json> --out OUT_G5_STAGE1
```
통과 조건(자동): 3방법 전 실행 OK · 행 완결 · 예열 OK · 방법별 VRAM 측정 존재. 통과 시 `OUT_G5_STAGE1/STAGE1_PASS.json`이 생긴다. 생기지 않으면 **Stage 2를 실행하지 말고** `summary.json`, `rows.jsonl`을 보낸다(진단 후 수정 → Stage 1부터 재실행).

## 2. Stage 2 — G3와 같은 BraTS 20건, 약 1–1.5시간 예상(추정)
```
mkdir OUT_G5_STAGE2
python3 -I tests/collect_env.py --out OUT_G5_STAGE2/environment.json --files <G3 summary.json>
python3 -I comparison/run_g5_v2.py --backend cupy --stage 2 --stage1-dir OUT_G5_STAGE1 \
   --runner-dir ... --repo ... --lib ... --brats-root ... --metadata ... \
   --g3-summary <gate_4090_n20_20261022_180342/summary.json> --out OUT_G5_STAGE2
```
코드 SHA-256 또는 GPU 이름이 Stage 1과 다르면 시작하지 않는다.

## 3. 돌려줄 파일
`environment.json`, `rows.jsonl`, `summary.json`, `summary.partial.json`, `subject_table.csv`, `run.log`(터미널 출력 `| tee`), Stage 1은 `STAGE1_PASS.json`. 실행 중 GPU를 다른 용도로 쓰지 않는다. 창을 닫거나 중단하면 `rows.jsonl`이 불완전해지므로 새 폴더에서 처음부터 다시 실행한다.

## 4. 자동으로 나오는 것
`summary.json["decision"]["verdict"]`(사전등록 규칙의 판정), `speed`(피험자별 r, 중앙값, 95% CI), `noise`, `order_effect`, `g3_consistency`, 피험자별 `profile`(커널별 GPU ms·실행 횟수·스레드 수·sync), `memory`(VRAM 최대 증가량), `pile_trace.alloc/occupancy`(pile 할당·점유).
