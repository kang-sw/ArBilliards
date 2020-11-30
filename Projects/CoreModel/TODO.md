# TODO

## Recognition
### 개요
영상 처리 프로그램
- 이전 테이블 위치 활용해 분리된 테이블 컨투어 병합하기 
- 마커 인식률 개선 & 노이즈 줄이기
- 공 위치 스무딩 및 인식률 높이기
- (큐대 인식하기 ... 가능하면!)

## BilliardPhysicsCore
- 피직스 시뮬레이션 모듈 구현
- API:
	PlaceBall(index, pos)
	HitBall(index, pos) returns TraceRecord
		struct TraceRecord {
			paths: PathNodeSet[]
			events: EventNode[]
			
			struct PathNodeSet{
				
			}

			struct EventNode{
				a, b: ColliderInfo
				contactPos: Vec3

				struct ColliderInfo {
					index : Int
					Pos : Vec3
					Vel : Vec3
					AngVel : Vec3
				}
			}
		}

## Unity

- 기존에 구현한 간단한 물리엔진은 공의 다음 경로 예측 용도로만 활용
	- 다음 경로 예측을 위해 각 패킷에 기록된 지연 시간으로 위치 보상
- 경로 정보 외부에서 받아오도록 코드 수정하기
- 공 사이의 충돌 이펙트 추가
