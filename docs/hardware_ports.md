# 현장 데스크톱 USB 연결 (2026-10-09, 경북대 현장)

데스크톱 USB 컨트롤러 하나(Intel `00:14.0`)에 모두 연결돼 있다. 포트 경로는 `버스-포트` 형식이고,
같은 물리 단자에 다시 꽂으면 같은 경로가 나온다.

| 장치 | 포트 경로 | 속도 | 리눅스 이름 | 고르는 기준 |
|---|---|---|---|---|
| RealSense **D435i** (위쪽, 테이블 내려다봄) | `2-3` | 5000 Mbps (USB 3) | `/dev/video0~5` (꽂는 순서에 따라 바뀜) | 시리얼 `261822074796` |
| RealSense **D405** (손목, 그리퍼에 장착) | `2-4` | 5000 Mbps (USB 3) | `/dev/video6~11` (바뀜) | 시리얼 `261122276195` |
| PiPER CAN 어댑터 (candleLight, gs_usb) | `1-6` | 12 Mbps | `can0` | 어댑터가 하나뿐이라 `can0` |
| Wi-Fi 동글 (Realtek) | `1-2` | 480 Mbps | `wlx84fc140c44ce` | — |
| USB 2.0 허브 (Prolific) → 키보드·마우스·SD 리더 | `1-1` (`1-1.1~3`) | 480 Mbps | — | — |

## 주의

- **카메라는 USB 3 단자에 PC 본체로 직접.** `1-1` 허브는 USB 2.0이라 D405가 아예 인식되지 않았다.
  꽂은 뒤 `cat /sys/bus/usb/devices/2-3/speed` (D405는 `2-4`)가 `5000`이면 정상, `480`이면 USB 2로 붙은 것.
- **카메라 시리얼은 `pyrealsense2` 값으로 고른다.** `lsusb -v`에 나오는 USB 시리얼(261643069213, 261723071075)은 다르다.
  `pac2026/config.yaml`의 `camera.top.serial` / `camera.wrist.serial`이 이 값이다.
- **D405는 깊이·컬러 해상도를 같게.** 다르면 pyrealsense2가 시작을 거부한다. 깊이 단위도 D405는 0.1 mm(D435i는 1 mm).
- 같은 카메라를 두 프로그램이 동시에 열 수 없다(뷰어 `perception-main/rs_dual_view.py`를 켜 둔 채 파이프라인 실행 X).

## 한 번에 확인

```bash
lsusb | grep -E "8086|1d50"          # 8086:0b3a D435i, 8086:0b5b D405, 1d50:606f CAN
for p in 2-3 2-4 1-6; do echo "$p $(cat /sys/bus/usb/devices/$p/product) $(cat /sys/bus/usb/devices/$p/speed)Mbps"; done
ip -details link show can0           # PiPER: state UP, bitrate 1000000 이어야 함
```
