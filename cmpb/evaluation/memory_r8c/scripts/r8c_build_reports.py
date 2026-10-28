"""Regenerate static audit reports from recorded E6 evidence and frozen code.

Usage: python -B scripts/r8c_build_reports.py
Does not execute GPU code. m=2/4 validation remains NOT MEASURED until server runs.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path
from r8c_common import HERE, readj, writej
from r8c_memory_model import model, build, extent, slab_shapes


def save(name,text): (HERE/'reports'/name).write_text(text.strip()+'\n',encoding='utf-8')


def main():
    a=readj(HERE/'results/evidence_audit.json'); e=a['e6']
    oldpath=HERE/'frozen/NEUROCOMPUTING_V1_INTEGRITY_RECOVERY/20261013/R8_SCALING_FEASIBILITY/ddfp_mem_model.py'
    sp=importlib.util.spec_from_file_location('r8a_frozen_model',oldpath); old=importlib.util.module_from_spec(sp); sp.loader.exec_module(old)
    comparison=[]
    for m in (1,2,4):
        shape=(240,240,155*m)
        for K in sorted(set((1,16,16*m))):
            frozen=old.k1(shape) if K==1 else old.kk(shape,K)
            new=model(shape,K); scenario=model(shape,K,reserve_factor=1.8,reserve_fixed_mib=256)
            comparison.append({'m':m,'K':K,'frozen_live_mib':frozen['vram']/2**20,'r8c_visible_live_mib':new['live_visible_peak_mib'],
                               'admission_reserved_scenario_mib':scenario['pool_reserved_prediction_bytes']/2**20,
                               'expanded_cells':new['expanded_cells'],'padded_cells':new['padded_cells'],
                               'global_verifier_blocks':new['global_verifier']['quantities']['total_blocks'],
                               'global_verifier_int32_ok':new['global_verifier']['ok'],
                               'observed_pool_mib':e['methods']['gpu_k1' if K==1 else 'ddfp_k16_d1']['median_subject_median_pool_mib'] if m==1 else None,
                               'validation_status':'CALIBRATION_SCALE_ONLY_NOT_LIVE_VALIDATION' if m==1 else 'NOT MEASURED'})
    writej(HERE/'results/R8C_MEMORY_MODEL_VALIDATION.json',{'kind':'STATIC_COMPARISON_WITH_RECORDED_M1_POOL_ONLY',
           'original_SC_M':e['SC_M'],'model_generalization_status':'NOT MEASURED','rows':comparison,
           'calibration_warning':'m1 pool/live ratios are scale-specific descriptions; not validated allocator laws',
           'internal_workspace_status':'CuPy reduction workspaces and rounding are not statically identified'})
    table='|m|K|Expanded cells|R8A frozen live MiB|R8C visible live MiB|Admission reserve scenario MiB|New live measurement|\n|---:|---:|---:|---:|---:|---:|---|\n'
    table+='\n'.join(f'|{r["m"]}|{r["K"]}|{r["expanded_cells"]:,}|{r["frozen_live_mib"]:.3f}|{r["r8c_visible_live_mib"]:.3f}|{r["admission_reserved_scenario_mib"]:.1f}|NOT MEASURED|' for r in comparison)
    # Exact array footprints for the anchor and every K16 halo slab.
    ledger=[]
    for label,shape in [('K1',(240,240,155))]+[(f'K16_slab{k:02d}',s) for k,s in enumerate(slab_shapes((240,240,155),16,1))]:
        npad=math.prod(extent(shape,True)); ne=math.prod(extent(shape)); w,h,d=shape
        for n,dt,sh,bytes_,phase in [
          ('s','float32',shape,4*math.prod(shape),'build'),('U_lo/U_hi each','float32',extent(shape),4*ne,'build'),
          ('lo/hi last face locals each','float32',(w,h-1,d-1),4*w*(h-1)*(d-1),'build'),
          ('c corner stack','float32',(8,w-1,h-1,d-1),32*(w-1)*(h-1)*(d-1),'build'),
          ('U_lo_pad/U_hi_pad/u each','float32',extent(shape,True),4*npad,'fp'),
          ('discovered/processed/scheduled/frontier each','int32',extent(shape,True),4*npad,'fp'),
          ('seeds and is_seed each','float32/int32',extent(shape,True),4*npad,'seeded fp only')]:
            ledger.append({'case':label,'name':n,'dtype':dt,'shape':str(sh),'bytes_each':bytes_,'MiB_each':bytes_/2**20,'phase':phase})
    with (HERE/'results/allocation_ledger.csv').open('w',encoding='utf-8',newline='') as f:
        wri=csv.DictWriter(f,fieldnames=list(ledger[0])); wri.writeheader(); wri.writerows(ledger)
    save('R8C_EVIDENCE_AUDIT.md',f'''# R8-C evidence audit — 2026-10-16

직접 읽은 근거는 `results/evidence_audit.json`의 원본 경로·SHA256·manifest별 대조 결과이다. 아래 수치는 기존 실측 기록의 독립 재집계이며 새 GPU 실측이 아니다.

|검사|독립 재계산|
|---|---|
|E1/E2/E6에 기록된 판정|각 PASS|
|단계별 artifact 해시|E1 {a['stages']['e1']['manifest_entries']}, E2 {a['stages']['e2']['manifest_entries']}, E6 {a['stages']['e6']['manifest_entries']}개, 모두 일치|
|E6 행|타이밍 {e['n_timed']} + 메모리 {e['n_mem']} = {e['n_total']}|
|실험 설계|20명 × 3반복 × 3방법, 중복·누락 없음|
|원본 production 해시|R7 EXPECTED 6개 모두 일치|
|E6 실행 코드 해시|r7_e6.py 및 r7_common.py가 run_meta 기록과 일치|
|E1↔E6 출력 해시|모든 타이밍 행 일치|
|K16 R*|2: 60/60|
|K1↔K16 출력|20명 모두 SHA256이 다름|
|전역 DWC|각 GPU 방법 60개 타이밍 기록에서 위반 합 0, PASS 기록|

타이밍은 먼저 피험자 내 3회 평균을 내고 그 20개 평균의 평균을 계산했다.

|방법|평균 s|피험자 평균 간 SD s|피험자별 pool 중앙값의 중앙값 MiB|
|---|---:|---:|---:|
'''+'\n'.join(f'|{m}|{v["mean_subject_mean_s"]:.7f}|{v["sd_subject_mean_s"]:.7f}|{v["median_subject_median_pool_mib"]:.1f}|' for m,v in e['methods'].items())+f'''

DD/K1의 피험자별 평균시간 비율을 20명에서 평균하면 {e['mean_paired_ratio_dd_over_k1']:.9f}; 전체 평균의 비율은 {e['ratio_of_means_dd_over_k1']:.9f}이다. 두 집계는 다른 통계량이다. 정당한 모집단 일반화와 인과 비교는 이 재집계만으로 얻어지지 않는다.

SC-M은 R8-A 및 R8-B A1의 기존 규칙 그대로 **FAIL (F4_MODEL_OUT_OF_BAND)**: K1 {e['SC_M']['measured_over_frozen']['k1']:.6f}배, K16 {e['SC_M']['measured_over_frozen']['k16']:.6f}배, pool 비율 {e['SC_M']['ratio']:.9f}>1.60이다. K1은 상한 2689 MiB 초과, K16은 [1284,1997] 안이다. R7 E6의 측정 유효성 PASS와 다른 판정이다.

Pool 수치는 `r7_e6.py` 125–130행의 알고리즘 직후·검증 전 타이밍 행이다. `mem` 행은 별도 샘플러이며 pool 항목이 없다. 메모리 패스 device delta는 K1 2986.0, K16 1840.0 MiB; 기준 device used는 두 방법 모두 429.1 MiB이다. `CupyBackend.memstats`는 device 전체 상태를 읽으며 개별 프로세스 VRAM과 동일하지 않다.

ZIP 상태는 JSON의 `zip_audit`에 별도로 기록했다. 사용자 지정 `v1/ddfp_r7_runs/r7_20261014_180628/e6.zip`은 현재 파일시스템에서 **MISSING_AT_AUDIT**이다. 각 ZIP의 현재 존재 여부·CRC 결과는 JSON의 개별 항목을 따른다. 원자료에 없는 ZIP을 검증 완료로 간주하지 않는다. 확인 가능한 runner/package ZIP에는 최신 E6 rows가 없었다. 따라서 최신 E6 ZIP 내용 검증은 완료하지 않았다. 실행 폴더의 49개 manifest 해시 일치는 별도로 확보했다.

선택 규칙은 E6 K1 피험자 내 중앙시간의 최소·하위 중앙·최대, 동률 subject ID이다. 선택값: {', '.join(e['selected_subjects_min_lower_median_max'])}. 모집단 무작위 표본이 아니다. BraTS 원 NIfTI와 GPU 출력 배열은 이 로컬 작업에 제공되지 않아 원영상에서의 재실행·DWC 재연산은 아직 하지 않았다.
''')
    save('R8C_MEMORY_ALLOCATION_TRACE.md',f'''# C1 allocation trace and C2 model

근거 소스는 `frozen/v1/DD-FP-R7/`의 해시 동결 사본이다. 행별 원문은 `evidence/numbered/v1/DD-FP-R7/`에 있다. 정확한 배열 footprint는 `results/allocation_ledger.csv`; 재계산 모델은 `scripts/r8c_memory_model.py`이다.

N0=WHD, Ne=(2W−1)(2H−1)(2D−1), Np=(2W+1)(2H+1)(2D+1), f=W(H−1)(D−1), c=(W−1)(H−1)(D−1). 모든 GPU 대형 배열은 float32 또는 int32이며 원소당 4 B이다. anchor Ne=70,897,269, Np=71,953,271. 하나의 padded float32 배열은 {4*71_953_271/2**20:.6f} MiB이다.

|경로·소스 행|배열/shape/dtype|생성과 참조 종료|동시 live 계산·후속 측정|
|---|---|---|---|
|gpu_immersion 57–60|s: (W,H,D) f32; U_lo/U_hi: expanded f32|build에서 생성, 함수 반환 시 소멸|4N0+8Ne|
|78–91|_mm4의 lo/hi: 마지막 (W,H−1,D−1), f32|다음 다중 할당의 RHS 동안 이전 lo/hi 생존; 마지막 쌍은 반환까지 생존|8f; 중첩 min/max 임시는 표현식 수명|
|93–96|c: (8,W−1,H−1,D−1), f32|두 reduction 이후에도 이름 c가 함수 반환까지 유지|32c; reduction 내부 workspace는 GPU 계측 필요|
|107–110|U_lo_pad/U_hi_pad: padded f32|두 번째 pad 생성 시 위 배열이 함께 생존|build pad live=4N0+8Ne+8f+32c+8Np|
|572–608|out host, meta dict, `_`=host aux|wrapper의 `_`는 aux를 실제로 보유; wrapper 반환까지 살아 있음|디바이스 FP 반환 후 cp.asarray(out) 재업로드|
|gpu_seed_aware 250–255|unseeded dispatcher|boundary_seeds=None이면 canonical로 위임|K1이 seeded workspace까지 쓰는 것은 아님|
|gpu_native 314–336|lo/hi views, u f32, discovered/processed/scheduled/frontier i32|lo_f/hi_f는 contiguous ravel view; 별도 대형 복사로 세지 않음|7×4Np + 1084 B; 작은 배열 1024+32+4+12+4+4+4 B|
|gpu_native 317–321,224–263|lo/hi host f32; float64 integer-validation 임시|D2H endpoints, host validation 뒤 FP 내 host endpoints 유지|host validation visible peak≈32Np; GPU 원소가 host f64가 되는 것은 아님|
|gpu_native 445–491|out f32, discovered/processed bool, scheduled i32, 작은 aux|out/meta/aux host 반환; device solver locals는 반환 시 소멸|대형 D2H payload 4Np×(lo,hi,u,discovered,processed,scheduled)=24Np; 이후 H2D 4Np|
|gpu_seed_aware 129–136,267–295|seeds f32, is_seed i32, lo/hi/u/state|seeds_dev와 seeds_f는 caller seeds를 alias; seeded host mask/levels는 RAM|seeded GPU 9×4Ns+1084; host mask·expected levels·aux seed map을 별도 기록|
|experiment_DDFP_all 208,234–235|u_assembled/u_prev/u_new: expanded f32|round0 1개; 갱신 중 3개 unique global; 다음 round 할당 RHS에서 이전 값 생존|4Ne 또는 12Ne; u_assembled=u_new 이후 alias 중복 계산 금지|
|215–216,241–252|U_lo/U_hi/seeds/u_sub slab padded backing|RHS 평가 중 직전 slab local이 남음; interior u_sub view가 padded backing 전체 유지|build 잔류 12Nprev(round0),16Nprev(seeded); FP 직전 u_sub 4Nprev 유지|
|266|difference와 abs: expanded f32 두 개|cp.abs(u_new-u_prev).max() 표현식 종료 뒤 해제|20Ne + 16Nlast; max reduction 내부 workspace 미포함|
|274|global final D2H f32|함수 반환 이후 GPU locals 소멸|host 출력 4Ne; C.f32는 이미 contiguous f32이면 복사하지 않음|

K1 solver live=28Np+1084 B이다. seeded DD의 slab solve는 12Ne+36Ns+4Nprev+1084 B, 수렴 검사는 20Ne+16Nlast이다. 각 단계의 max가 tensor envelope이다. 정확한 allocator 배치·reduction workspace가 빠져 있어 device peak의 엄밀한 상계라고 부르지 않는다.

ELI5: 작업대 위에 지금 쓰는 종이(live)와 서랍에 다시 쓸 수 있게 보관한 종이(pool reserved)는 다르다. 작업을 끝내도 서랍이 비워지는 것은 아니다. 이 비유는 지표의 차이를 설명하며 실제 초과분의 원인을 특정하지 않는다.

마지막 face locals {8*240*239*154/2**20:.6f} MiB가 R8-A build 식에서 빠져 있다. 그러나 이를 추가해도 K1은 FP, K16은 convergence가 peak이므로 최종 예측이 사실상 그대로다. **이 수정만으로 약 1 GiB 차이가 해결되지 않는다.**

{table}

R8C reserve 시나리오는 live×1.8+256 MiB이며 자원 admission용 가정이다. 과학적 성공 범위나 경험 법칙이 아니다. m=1 pool/live 보정배수는 K1 약 1.5444, K16 약 1.2518이고 m=2/4 재사용의 타당성은 미실측이다.

P(t)=used_bytes(t)+free_bytes(t), D(t)=P(t)+Xcontext/library(t)+Oother_process(t)로 구분한다. used_bytes는 할당 블록량이며 rounding을 포함할 수 있어 정적 tensor bytes와 동일하지 않다. E6는 algorithm live peak를 기록하지 않았으므로 max(P)−max(Lmodel)를 모두 단편화라고 부를 수 없다. 서로 다른 시각의 peak 차감도 인과 분해가 아니다. `r8c_gpu_memory_probe.py`는 동일 시각의 pool used/total/device와 nvidia-smi 프로세스 샘플을 분리한다.

원인 후보와 검증 실험: (1) build의 c 및 unpadded endpoints가 풀에 남는 효과는 build/FP 전환 timeline과 allocator endpoints로 확인, (2) slab 크기별 allocator 재사용은 동일 K에서 단계별 total 증가·free cache 확인, (3) reduction 내부 공간은 convergence 전후·할당 이벤트 확인, (4) context/기타 프로세스는 프로세스별 inventory와 사전 baseline 확인. pool purge 또는 allocator 변경 실험은 별도 버전의 메타데이터 대조군으로만 설계하고 본 타이밍과 섞지 않는다. 원인별 인과 기여율은 **NOT MEASURED**.

CuPy allocator 의미의 출처: [CuPy 14.2.0 Memory Management](https://docs.cupy.dev/en/v14.2.0/user_guide/memory.html), [MemoryPool](https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.cuda.MemoryPool.html). 문서는 caching/rounding/context를 설명하며 이 연구의 초과분 분해를 입증하지 않는다.
''')
    save('R8C_SCALING_PROTOCOL.md',f'''# R8-C V1 scaling protocol — fixed before new GPU results

상태: 코드 준비·CPU 검증 단계. 사용자가 RTX 4090 서버에서 직접 실행하고 결과를 전달하기로 했다. GPU smoke 및 primary scaling은 **NOT EXECUTED**. R8-A와 R8-B 규칙·원본 파일은 변경하지 않았다.

선택 피험자: {', '.join(e['selected_subjects_min_lower_median_max'])}. E6 K1 피험자별 3회 중앙시간의 최소·하위 중앙·최대, 동률 ID. m=1,2,4이며 홀수 번째 추가 copy는 z 반전한다(R7 tile_z와 동일). K1, K16, K16m, δ=1, max_rounds=8. m=1 K16m은 K16 alias이며 재실행하지 않는다.

각 subject×m×method에 timing 3개 + metadata 1개, 모두 독립 프로세스. 기본 순서는 seed=20261016으로 size 내 subject/method 조합을 순열화한다. warm-up은 tiny original path로 각 프로세스에서 별도 수행하고 pool release 뒤 시작한다. 총 primary 96프로세스 + smoke 4프로세스. m=1 E6는 기초 자료와 회귀 해시로 재사용하며, 새 live/reserved 비교의 프로세스 조건을 맞추기 위해 재측정한다. E6 시간과 새 시간을 하나의 평균으로 혼합하지 않는다.

필수 순서: CPU tests → 원본·환경·입력 해시 확인 → GPU tiny smoke K1/K16 각 2회(CPU parity, 독립 CPU DWC 및 주입된 opposite-corner 위반 검출) → primary → 재검증 → 분석. README의 명령을 사용한다.

최초 실행부터 live×1.8+256 MiB reserve 시나리오가 device 예산 min(0.75×total−baseline,total−baseline−3072MiB)에 들어야 한다. host 예산은 available×0.6, 예측은 tensor host scenario+32Ne+1GiB의 별도 보수적 가정이다. 이전 metadata 관측이 있으면 1.15×관측값×m비율도 추가 하한으로 적용한다. 이는 OOM 불가능성의 증명이 아니다. 기본 m4 K1 reserve 가정은 약 13.7GiB(14,056MiB)이다. 전체 저장 .npy 및 2GiB 여유의 사전 disk gate를 적용한다.

algorithm timing은 동기화한 CPU wall-clock phase 합이다. CUDA-event envelope는 host wait와 전송도 포함하므로 kernel time으로 부르지 않는다. pure kernel time=NOT MEASURED. metadata 패스만 allocator wrapper·원본 함수 AST 사본의 scalar checkpoint·10ms RSS/device sampling·0.5s process inventory·R_act per-solve 수집을 수행한다. pool used/total high는 allocator endpoints와 stage snapshots, non-pool/device/RSS high는 sampled lower bound로 표시한다. HWM reset 실패 시 process HWM과 algorithm sampled RSS를 구분한다. metadata 시간은 primary 성능 통계에서 제외한다.

모든 출력은 finite·shape/dtype·원 voxel 보존·원 Ispan interval(모든 셀, host chunked)·전역 face/cube DWC·seam 검사를 받는다. m1은 E1 full SHA256과 회귀 비교; m2/4 기준 출력은 없으므로 regression NOT_AVAILABLE이다. 타이밍 3회와 metadata의 full-output SHA256이 같아야 한다. 전역 검사 불가 크기에서는 REFUSED_INT32, PASS key를 생성하지 않는다. m8 total blocks=2,267,330,860으로 거부된다. m6/8 자동 실행은 지원하지 않으며 후속 사전등록과 gate가 필요하다.

R*>2는 자동 실패가 아니다. R*<8인 정지 확인과 per-solve 수 K(1+R*)를 검사하고 metadata에서 실제 max_change를 기록한다. R*=8은 마지막 round 수렴 여부를 timing 경로에서 구별할 수 없어 NOT_CONFIRMED로 보수 처리한다. 이를 실제 비종료라고 단정하지 않는다.

OOM, timeout(각 worker 1800s), 다른 GPU compute pid, 출력 비결정성, DWC/seam/interval/회귀 위반은 전체 controller를 중단한다. 기록·로그·부분 결과를 보존한다. STOPPED 폴더는 덮어쓰지 않는다. 원인 수정 후 새 폴더를 사용한다. 정상 완료 worker만 같은 fingerprint에서 재사용 가능하다. source/config/input hash가 다르면 resume하지 않는다.

분석은 피험자 내 3회 평균·중앙값·SD를 먼저 내고, 피험자 간 변동을 분리한다. DD/K1 ratio와 log ratio~log m의 피험자별 slope, 선택된 3사례의 bootstrap 재표집 범위를 기술한다(B=10000, seed20261016). 이는 모집단 CI가 아니다. crossover는 m별 실측 DD/K1<1로 관찰하고 slope 감소와 구분한다. R_act·slab padded cells·solver 개수·transfer payload·phase time을 함께 비교하되 인과성은 통제 실험 없이는 확정하지 않는다.

새 모델은 m1 보정과 m2/4 평가를 구분하며, 기존 SC-M FAIL을 변경하지 않는다. 새 규모 검증의 최종 기여 판정은 현재 INCONCLUSIVE. 자동 분석의 descriptive 결과 뒤 연구자가 보장 범위·효율·영상 성질을 분리해 평가한다.
''')
    save('R8C_SCALING_RESULTS.md',f'''# R8-C scaling results

새 GPU smoke: **NOT EXECUTED**. m=1/2/4 primary: **NOT MEASURED**. 사용자 직접 서버 실행 예정. 로컬 GPU는 RTX 3080 Ti Laptop 16GiB이고 요구 서버는 RTX 4090이다. 로컬 runtime에 CuPy와 BraTS가 없어 fail-closed preflight를 확인했다(`results/local_preflight/preflight.json`). 이 실패는 알고리즘 성능의 실패가 아니다.

기존 E6 재계산: K1 {e['methods']['gpu_k1']['mean_subject_mean_s']:.7f}s, DD16 {e['methods']['ddfp_k16_d1']['mean_subject_mean_s']:.7f}s, CPU Sequential {e['methods']['seq_fp_cpu']['mean_subject_mean_s']:.7f}s. pool reserved 2967.4/1785.6MiB, sampled device delta2986/1840MiB. 기존 규모의 DD16 pool 예약량은 K1보다 약 {(1-1785.6/2967.4)*100:.3f}% 작다. 이것을 live peak 감소율이라고 부를 수 없다. 피험자별 DD/K1 평균 비율은 {e['mean_paired_ratio_dd_over_k1']:.9f}이다. 신규 scaling의 숫자는 없다.

실행 후 `r8c_validate.py --run ...`와 `r8c_analyze.py --run ...`로 새 결과를 생성한다. 개별 worker와 환경·admission·hash·saved output이 모두 있어야 PASS; 불완전 기록은 통계로 승격하지 않는다.
''')
    save('R8C_SCIENTIFIC_ASSESSMENT.md',f'''# Scientific assessment

전체 후속 연구 판정: **INCONCLUSIVE**. 기존 E6에서 작은 pool 예약량은 관찰되지만 확대 규모와 live peak는 미측정이다. 사전등록 SC-M: **FAIL 유지**. 새로운 알고리즘 최적화나 V1/V2는 구현하지 않았다.

|질문|확인된 근거|판정·남은 검증|
|---|---|---|
|Q1 정적 모델 불일치|live tensor model과 pool reserved가 다른 양이며 build face locals 누락 확인|누락 보정 뒤 peak는 거의 동일. 캐시/rounding/workspace/context별 기여율 미확정|
|Q2 DD 메모리 이득|E6 pool 1785.6 vs2967.4MiB, device delta1840 vs2986MiB|실측 pool/device 절감. live vs cached 구성 분해 NOT MEASURED|
|Q3 규모별 시간 비율|E6 m1 paired mean DD/K1={e['mean_paired_ratio_dd_over_k1']:.9f}|m2/4 미측정, crossover 미확정|
|Q4 계산 메커니즘|K1 1solve, DD K(1+R*) solve; 작은 slab와 global convergence scan 공존|R_act·padded-cell work·host transfer·synchronization 기여는 metadata 비교 필요|
|Q5 출력 차이|20/20 K1/DD hash 차이, 각 기록 전역 DWC0; DD에는 pinned seeds·slab 독립 이력·조립 순서가 있음|계산 경로 차이는 코드 사실. 특정 차이 voxel의 원인 분해는 가설; 이 코호트의 다른 두 출력이 모두 기록된 DWC 검사를 통과함|
|Q6 구조 한계|현재 DD에는 3 global f32 상태와 convergence 두 temporary가 있음|visible convergence20Ne floor; global state 제거의 이득은 잠재적 설계 가설|

증거 유형: ①기록: R7 verdict·rows·run_meta, ②독립 재계산: counts·통계·hash·배열 byte, ③코드 사실: seed/pin 및 allocation lifetime, ④정의로부터 산술 유도: 28Np/20Ne 계수, ⑤미측정 가설: allocator 구성별 초과분·scaling 인과성, ⑥미검증 외삽: streaming/compact capacity gain. 새로운 일반정리나 임상 품질 개선은 이 자료에서 도출하지 않는다.

수학적 범위: 원 Ispan singleton인 voxel의 값은 interval 보존과 함께 보존된다(구간[a,a]의 원소는 a뿐). 이것은 구간 보존이 실제 구현에서 성립한다는 검증과 분리해야 한다. 새 러너는 모든 original interval을 독립 재계산해 검사한다. ELI5: 선택 가능한 색이 한 가지뿐인 칸은 그 색을 유지해야 한다.

전역 DWC 인증은 구현한 face/cube criterion의 모든 대상 block을 유효 인덱스로 검사한 출력에 한정한다. seam xz/yz 검사는 전역 검사의 대체물이 아니다. R*=2는 기존 코호트 경험적 결과이며 모든 입력의 필요조건·충분조건이나 종료 정리로 격상하지 않는다. production docstring의 'Thm.1'·출력 동등성 주석은 독립 증명으로 인용하지 않는다.

영상 품질·위상·DWC·원 voxel 보존은 별개이다. 서로 다른 두 출력이 모두 DWC를 만족해도 영상 충실도, 분할 정확도, 임상 효용 또는 원본 대비 위상 보존의 우열을 결정하지 못한다. 여기서는 그런 평가값을 생성하지 않았다.

|후속 후보|착수 전에 필요한 근거|
|---|---|
|GPU-native FP 최적화|host validation/aux transfer 비용의 실측 분해, 같은 출력 회귀|
|IBI scheduler·분할 정책|R_act vs slab 크기, halo 중복, R* 추세, 원조건 유지|
|V1 compact global state|20Ne floor 지배의 실측, datatype/alias 변경의 정확성 의무|
|V2 host-staged streaming|host 전송 예산·global state 제거와 수렴/seed provenance 증명|
|local/seam 완전 인증|모든 face/cube 및 owner-crossing block의 완전한 coverage 증명·음성 대조|

선행 allocator 의미는 [CuPy14.2.0](https://docs.cupy.dev/en/v14.2.0/user_guide/memory.html)에 근거한다. 수치·구현 주장은 본 패키지의 동결 코드와 evidence JSON에 근거한다. 새로운 DWC 선행연구 우선성 또는 보장 확대 주장은 하지 않는다.
''')
    writej(HERE/'results/execution_status.json',{'C1':'CODE_TRACE_COMPLETE_CAUSAL_DECOMPOSITION_PENDING_GPU','C2':'STATIC_MODEL_READY_NOT_VALIDATED_OUT_OF_SCALE',
           'C3':'CPU_CHECKS_COMPLETE_GPU_SMOKE_NOT_EXECUTED','C4':'NOT EXECUTED_USER_WILL_RUN_SERVER','C5':'INCONCLUSIVE',
           'frozen_SC_M':'FAIL','reason':'User requested server-ready code and will supply actual RTX4090 results.'})
    print('reports and static model validation generated; no GPU measurements')


if __name__=='__main__': main()
