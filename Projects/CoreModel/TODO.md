# TODO

## Recognition
### 개요
영상 처리 프로그램
- 테이블 위치 팝핑 너무 심함
- 공 인식 잘 안 됨(색공간 바꾸기?)
- 마커 인식 노이즈 줄이기
	- 마커 solver GPU 이용하게 개선
- (큐대 인식하기 ... 가능하면!)

## BilliardPhysicsCore
- 피직스 시뮬레이션 모듈 구현
- Boost.Python으로 호출 가능한 모듈 만들기, Pybullet 기반 평가 모듈
	- 평가 모듈 API: PlaceBall(index, pos), EvalBall(indesc, power, phi, xshift)

## Unity

- 기존에 구현한 간단한 물리엔진은 공의 다음 경로 예측 용도로만 활용
	- 다음 경로 예측을 위해 각 패킷에 기록된 지연 시간으로 위치 보상
- 경로 정보 외부에서 받아오도록 코드 수정하기
- 공 사이의 충돌 이펙트 추가
