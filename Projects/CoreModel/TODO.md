# TODO

## Recognition
### ����
���� ó�� ���α׷�
- ��Ŀ �ν� ������ ���̱�
	- ��Ŀ solver GPU �̿��ϰ� ����
- (ť�� �ν��ϱ� ... �����ϸ�!)

## BilliardPhysicsCore
- ������ �ùķ��̼� ��� ����
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

- ������ ������ ������ ���������� ���� ���� ��� ���� �뵵�θ� Ȱ��
	- ���� ��� ������ ���� �� ��Ŷ�� ��ϵ� ���� �ð����� ��ġ ����
- ��� ���� �ܺο��� �޾ƿ����� �ڵ� �����ϱ�
- �� ������ �浹 ����Ʈ �߰�
