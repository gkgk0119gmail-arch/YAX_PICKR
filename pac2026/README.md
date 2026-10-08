# PAC 2026 · 파티션 칸별 적재 파이프라인 (PICKR Lab)

실행 파일 하나(`run.py`)가 장비와 모델을 한 번 준비하고, 기본과제 1–7단계를 피드백 루프로 실행한다.
흐름도: `../diagrams/basic_task_flow.png`

## 구조
```
pac2026/
├── run.py            진입점 (CLI 옵션)
├── config.yaml       카메라·모델·임계값·로봇·모션 설정 (ON SITE 표시 값은 현장에서 측정)
├── calib/            pixel_to_robot.json (현장 캘리브레이션 결과, 없으면 MOCK 변환)
├── runs/             실행마다 로그·단계 이미지·json 저장
└── pac/
    ├── types.py      단계 간 데이터: Frame, BoxInfo, CellInfo, Partition, Grasp
    ├── cameras.py    상단 D435i (V4L2) / 오프라인 이미지. D405 손목 카메라는 TODO
    ├── models.py     모델 1회 로딩: Grounding DINO, VGGT-1B-Commercial, RF-DETR+SAM2, DINOv3 분류기
    ├── perception.py 단계 함수: scan_box, split_partition, find_blocks, cell_state
    ├── calib.py      픽셀 → 로봇 XY 호모그래피, 이미지 각도 → 로봇 yaw
    ├── robot.py      MockRobot / PiperRobot(piper_sdk, 현장 검증 전)
    ├── pipeline.py   1–7단계 피드백 루프 (재시도·되돌아가기)
    ├── runlog.py     로그 폴더 + 대시보드 전달
    └── dashboard.py  모니터링 대시보드 (파이썬 표준 웹 서버, http://localhost:8800)
```
비전 코드는 아직 `../perception-main/` 에 있고 `pac/paths.py` 로 불러온다.

## 실행
```bash
cd pac2026
PY="HF_HUB_OFFLINE=1 ../perception-main/ai_bubble/.venv/bin/python"

# 오프라인 사진 + mock 로봇 (지금 바로 가능)
$PY run.py --image ../perception-main/data/vggt/cam4.jpg --depth ../perception-main/data/vggt/depth.raw
$PY run.py --image ... --mock-fail-rate 0.4      # 집기 실패를 섞어서 재시도 확인
$PY run.py --image ... --step partition          # 2단계까지만

# 실시간 D435i + mock 로봇
$PY run.py

# 실제 PiPER (CAN 활성화 후, config ON SITE 값 확인 후)
$PY run.py --robot piper
```

## 모니터링 대시보드
`run.py` 를 실행하면 자동으로 켜진다 → 브라우저에서 **http://localhost:8800** (같은 네트워크의 노트북: 터미널에 찍히는 LAN 주소).
- 실시간 화면: 카메라 + 인식 오버레이(상자, 칸 상태 색, 목표 칸, 블록·그리퍼 방향, 현재 단계)
- 단계 진행 1–7 (성공 ✓ / 실패 ✗ 횟수), 핵심 수치(상자 높이, n×m, 빈 칸, 그리퍼 폭, 재시도)
- 모델 카드: Grounding DINO · VGGT 높이 · Canny+RANSAC · DINOv3 칸 판정 · RF-DETR+SAM2+RANSAC · 넣은 뒤 판정
- 모델별 처리 시간, 실시간 로그, 지난 실행 목록, **비상 정지 버튼**(다음 확인 시점에 로봇 정지)
- 실행이 끝나도 대시보드는 열려 있다(Ctrl+C로 종료). `--no-dashboard`, `--port`, `--no-hold` 옵션.

## 단계 ↔ 함수
| 단계 | 함수 | 상태 |
|---|---|---|
| 1 상자 위치·높이 | `perception.scan_box` | ✅ |
| 2 파티션 n×m·빈 칸 | `perception.split_partition` | ✅ |
| 3 블록 찾기·집기 | `perception.find_blocks` → `Pipeline.step3_pick` | ✅ 비전 / 🔶 로봇(mock) |
| 4 파지 확인 | `Pipeline.step4_holding` (그리퍼 폭) | 🔶 D405 확인 TODO |
| 5 목표 칸 위 | `Pipeline.step5_move_above` | 🔶 |
| 6 위치·파지 재확인 | `Pipeline.step6_check` | 🔶 위치 확인·D405 정렬 TODO |
| 7 내려놓기·판정 | `Pipeline.step7_place_and_verify` + `perception.cell_state` | ✅ 판정 / 🔶 로봇 |

## 현장에서 할 일
1. `config.yaml` ON SITE 값: `tcp_offset_m`, `down_rpy_deg`, `table_z_robot_m`
2. 캘리브레이션 → `calib/pixel_to_robot.json` (`PixelToRobot.fit` 사용, 점 4개 이상)
3. `--robot piper` 로 천천히(speed_percent 낮게) 1블록부터

## 모델·라이선스
가중치는 저장소에 넣지 않는다. Hugging Face에서 받거나(승인제: DINOv3, VGGT-1B-Commercial) 팀 USB 사본 사용.
우리가 학습한 RF-DETR(`checkpoint_best_total.pth`)·DINOv3 칸 분류기는 Release로 배포.
