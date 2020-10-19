using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net;
using UnityEngine;

namespace ArBilliards.Phys
{
	public static class Constants
	{
		public const float KINDA_SMALL_NUMBER = 1e-7f;
		public const float G = 9.8f;
	}

	public class PhysContext
	{
		// 10밀리초 이내에 연산이 끝나지 않으면 타임아웃.
		public int TimeoutMs = 1000;

		public Vector3 UpVector = Vector3.up;

		public int Count => _objects.Count;
		public PhysObject this[int i] => _objects[i];
		public IEnumerable<PhysObject> Enumerable => _objects;

		private List<PhysObject> _objects = new List<PhysObject>();
		private List<HashSet<PhysObject>> _overlaps = new List<HashSet<PhysObject>>();
		private HashSet<int> _availableIndexes = new HashSet<int>();

		public void Clear()
		{
			_availableIndexes.Clear();
			_overlaps.Clear();
			_objects.Clear();
		}

		public int Spawn(PhysObject obj)
		{
			obj = (PhysObject)obj.Clone();
			obj.__InternalSetContext(this);
			int index;

			if (_availableIndexes.Count > 0)
			{
				index = _availableIndexes.GetEnumerator().Current;
				_objects[index] = obj;
			}
			else
			{
				index = Count;
				_objects.Add(obj);
				_overlaps.Add(new HashSet<PhysObject>());
			}

			return index;
		}

		public void Destroy(int index)
		{
			_objects[index] = null;
			_availableIndexes.Add(index);
			_overlaps[index].Clear();
		}

		public void ResetOverlapStates()
		{
			foreach (var overlap in _overlaps)
			{
				overlap.Clear();
			}
		}

		public struct ContactInfo
		{
			public float Time;
			public (int Idx, Vector3 Pos, Vector3 Vel) A, B;
			public Vector3 At;
		}

		/// <summary>
		///  시뮬레이션을 수행, 내부 오브젝트의 상태를 업데이트합니다.
		/// </summary>
		/// <param name="deltaTime"></param>
		/// <param name="result">결과가 저장됩니다.</param>
		public void StepSimulation(float deltaTime, List<ContactInfo> result)
		{
			result.Clear();

			float totalTime = 0.0f;
			const float SMALL_NUMBER = Constants.KINDA_SMALL_NUMBER;
			var timeoutTimer = new Stopwatch();
			timeoutTimer.Start();

			// 남은 시간이 0이 될 때까지 반복합니다.
			while (deltaTime > 0)
			{
				float minContactTime = deltaTime;
				(int A, int B)? minContactIdx = null;
				PhysObject.Contact? minContact = null;
				bool bOverlapOccured = false;

				// 가장 먼저 발생하는 이벤트를 찾습니다.
				foreach (var A in Enumerable)
				{
					minContactTime = Mathf.Min(minContactTime, A.CalcNextEventTime());
				}

				// 가장 먼저 접촉하는 페어를 찾습니다.
				for (int idxA = 0; idxA < Count; idxA++)
				{
					var A = this[idxA];
					var overlaps = _overlaps[idxA];

					if (A == null)
						continue;

					for (int idxB = idxA + 1; idxB < Count; idxB++)
					{
						var B = this[idxB];
						if (B == null)
							continue;

						var contact = A.CalcMinContactTime(B);

						if (!contact.HasValue)
						{
							overlaps.Remove(B);
							continue;
						}

						var ct = contact.Value;

						// 만약 오버랩이 검출되었고, 이미 오버랩 처리중이라면 돌아갑니다.
						if (ct.Time == 0)
						{
							// 한 틱에 한 번의 오버랩만 처리됩니다.
							if (bOverlapOccured)
								continue;

							if (ct.A.Vel.sqrMagnitude + ct.B.Vel.sqrMagnitude == 0 || overlaps.Contains(B))
								continue;

							bOverlapOccured = true;
							overlaps.Add(B);
						}

						if (ct.Time < minContactTime)
						{
							minContactIdx = (idxA, idxB);
							minContactTime = contact.Value.Time;
							minContact = contact;
						}
					}
				}

				// 타임아웃 검사
				if (timeoutTimer.ElapsedMilliseconds > TimeoutMs)
				{
					return;
				}

				// 최소 델타 시간만큼 모든 오브젝트를 전진시킵니다.
				foreach (var elem in Enumerable)
				{
					elem.AdvanceMovement(minContactTime);
				}

				totalTime += minContactTime;

				if (minContactIdx.HasValue)
				{
					var idx = minContactIdx.Value;
					var (A, B) = (this[idx.A], this[idx.B]);

					// 충돌을 계산합니다.
					A.ApplyCollision(B);

					// 충돌을 기록합니다.
					ContactInfo contact;
					contact.A = (idx.A, A.Position, A.Velocity);
					contact.B = (idx.B, B.Position, B.Velocity);
					contact.Time = totalTime;
					contact.At = minContact.Value.At;

					result.Add(contact);
				}

				deltaTime -= minContactTime;
			}
		}
	}

	public enum PhysType
	{
		Sphere,
		StaticPlane
	}

	public abstract class PhysObject : ICloneable
	{
		public abstract PhysType Type { get; }
		public PhysContext Context { get; private set; }
		public float Mass = 1.0f;
		public Vector3 Position { get; protected set; }
		public Vector3 Velocity { get; protected set; }
		public Vector3 AngularVelocity { get; protected set; }
		public double Damping = 0.001;
		public float Restitution = 1.0f;
		public (float Kinetic, float Rolling, float Static) Friction = (0.2f, 0.01f, 0.21f);

		public double DeltaSinceLastCollision { get; private set; }

		public Vector3 AngularVelocityInternal { set => AngularVelocity = value; }
		public Vector3 VelocityInternal { set => Velocity = value; }

		public Vector3 SourceVelocity
		{
			get => _sourceVelocity;
			set {
				DeltaSinceLastCollision = 0;
				_sourcePosition = Position;
				_sourceAngularVelocity = AngularVelocity;
				Velocity = _sourceVelocity = value;
			}
		}

		public Vector3 SourcePosition
		{
			get => _sourcePosition;
			set {
				DeltaSinceLastCollision = 0;
				Position = _sourcePosition = value;
				_sourceAngularVelocity = AngularVelocity;
				_sourceVelocity = Velocity;
			}
		}

		public Vector3 SourceAngularVelocity
		{
			get => _sourceAngularVelocity;
			set {
				DeltaSinceLastCollision = 0;
				_sourcePosition = Position;
				AngularVelocity = _sourceAngularVelocity = value;
				_sourceVelocity = Velocity;
			}
		}

		public Vector3 _sourceVelocity;
		public Vector3 _sourcePosition;
		public Vector3 _sourceAngularVelocity;

		/// <summary>
		/// AdvanceMovement가 명시적으로 호출되어야 하는 다음 순간을 반환합니다.
		/// 당구공의 경우, 미끄러지는 공이 구르기 시작할 때 마찰 계수를 계산하기 위해 사용합니다.
		/// </summary>
		/// <returns></returns>
		public virtual float CalcNextEventTime()
		{
			return float.MaxValue;
		}

		/// <summary>
		/// 현재 위치와 속도를 바탕으로 대상 오브젝트와 충돌하는 최소 시간을 구합니다.
		/// </summary>
		/// <param name="other"></param>
		/// <returns>충돌 정보입니다.</returns>
		public abstract Contact? CalcMinContactTime(PhysObject other);

		/// <summary>
		/// 현재 위치와 속도를 바탕으로 오브젝트를 전진시킵니다.
		/// 당연하지만 충돌 계산은 하지 않으므로, delta 값은 다른 오브젝트와의 contact 타임 중 가장 적은 값이 되어야 합니다.
		/// </summary>
		/// <param name="delta">다른 충돌체에 충돌하기까지의 시간보다 짧은 델타 시간입니다.</param>
		/// <returns></returns>
		public void AdvanceMovement(float delta)
		{
			DeltaSinceLastCollision += delta;
			AdvanceMovementImpl((float)DeltaSinceLastCollision);
		}

		protected virtual void AdvanceMovementImpl(float delta)
		{
			var eat = Math.Exp(-Damping * DeltaSinceLastCollision);
			Vec3d V = _sourceVelocity;
			Vec3d P = _sourcePosition;

			Velocity = V * eat;
			Position = P + V * (1.0 / Damping) * (1.0f - eat);
		}

		/// <summary>
		/// 두 물체 사이의 충돌 연산을 수행하고 물리 상태를 변화시킵니다.
		/// 위치는 고려하지 않으며, 속도와 질량 등 여러 물리 계수를 고려해 현재 속도를 변화시킵니다.
		/// </summary>
		/// <param name="other">연산을 적용할 대상 오브젝트입니다.</param>
		/// <returns></returns>
		public void ApplyCollision(PhysObject other)
		{
			ApplyCollisionImpl(other);

			_sourceVelocity = Velocity;
			_sourcePosition = Position;
			_sourceAngularVelocity = AngularVelocity;
			DeltaSinceLastCollision = 0;

			other._sourceVelocity = other.Velocity;
			other._sourcePosition = other.Position;
			other._sourceAngularVelocity = other.AngularVelocity;
			other.DeltaSinceLastCollision = 0;
		}

		public abstract void ApplyCollisionImpl(PhysObject other);

		public struct Contact
		{
			public (Vector3 Pos, Vector3 Vel) A, B;
			public float Time;
			public Vector3 At; // 충돌 자체가 일어난 지점
			public float OverlapDepth;
		}

		public object Clone()
		{
			return MemberwiseClone();
		}

		public void __InternalSetContext(PhysContext owner)
		{
			if (Context != null)
			{
				throw new Exception("Invalid access to internal method!");
			}
			Context = owner;
		}
	}

	internal class PhysSphere : PhysObject
	{
		public float Radius = 1.0f;
		public float RollBeginTime = 0.3f;

		public override PhysType Type => PhysType.Sphere;

		public override Contact? CalcMinContactTime(PhysObject other)
		{
			switch (other.Type)
			{
			case PhysType.Sphere:
				return CalcContact((PhysSphere)other);
			case PhysType.StaticPlane:
				return CalcContact((PhysStaticPlane)other);
			default:
				throw new ArgumentOutOfRangeException();
			}
		}

		public bool IsRolling()
		{
			var Vp = Velocity;
			var W = AngularVelocity;

			var sizeVp = Vp.magnitude;
			var RsizeW = Radius * W.magnitude;

			return RsizeW >= sizeVp;
		}

		protected override void AdvanceMovementImpl(float delta)
		{
			// 기본 이동을 적용
			base.AdvanceMovementImpl(delta);

			// 각속도 업데이트
			// 각속도의 최대치를 구하고, 시간을 바탕으로 각속도를 예측합니다.
			var maxAngVel = Vector3.Cross(Velocity, -Context.UpVector) / Radius;
			var angVelRatio = Math.Min((float)DeltaSinceLastCollision / RollBeginTime, 1.0f);

			var maxAngVelDelta = maxAngVel - AngularVelocity;
			AngularVelocity = _sourceAngularVelocity + maxAngVelDelta * angVelRatio;
		}

		public override void ApplyCollisionImpl(PhysObject B)
		{
			// 충돌 각도를 계산하기 위해, 먼저 두 구의 중심선에 대한 벡터 투영을 구합니다.
			// ref: https://sites.google.com/site/3dgameprogram/home/physics-modeling-for-game-programming/-gibongaenyeomdajigi---mulli/-jiljeom-ui-chungdol
			const float SMALL_NUMBER = Constants.KINDA_SMALL_NUMBER;
			var A = this;

			var (e, m1, m2) = (0.5f * (A.Restitution + B.Restitution), A.Mass, B.Mass);
			var (V1, V2) = (A.Velocity, B.Velocity);

			void ApplyAngularDelta(PhysSphere S, float friction, Vector3 contactDir)
			{
				S.Velocity += Radius * Vector3.Cross(S.AngularVelocity * friction, S.Context.UpVector);
				S.AngularVelocity *= (1.0f - friction);
			}

			switch (B.Type)
			{
			case PhysType.Sphere:
			{
				// 선형 성분을 계산합니다.
				var center = (B.Position - A.Position).normalized;

				var V1p = Vector3.Project(V1, center);
				var V2p = Vector3.Project(V2, -center);

				var nV1p = ((m1 - e * m2) * V1p + (1 + e) * m2 * V2p) / (m1 + m2);
				var nV2p = ((m2 - e * m1) * V2p + (1 + e) * m1 * V1p) / (m1 + m2);

				var nV1 = V1 + (nV1p - V1p);
				var nV2 = V2 + (nV2p - V2p);

				A.Velocity = nV1;
				B.VelocityInternal = nV2;

				var angle = Mathf.Abs(Vector3.Angle(V1 - V2, center));

				// 단순 회전 모델 = 회전을 정지 마찰력만큼 감쇠시키고 속도에 그대로 가산합니다.
				ApplyAngularDelta(A, (90f - angle) / 90f * A.Friction.Static, center);
				ApplyAngularDelta((PhysSphere)B, (90f - angle) / 90f * B.Friction.Static, -center);

				break;
			}
			case PhysType.StaticPlane:
			{
				// 반발 계수에 기반해 단순한 충돌을 계산합니다.
				var PL = (PhysStaticPlane)B;
				var N = PL.Normal;

				var V1p = Vector3.Project(V1, N);
				var nV1p = -e * V1p;

				A.Velocity = (nV1p - V1p) + V1;

				// 공이 노멀과 마주보게끔 합니다.
				if (Vector3.Angle(A.Velocity, N) > 90f)
					N = -N;

				// 공의 현재 회전 정도를 속도에 반영하고, 현재 속도와 평면의 관계로부터 회전에 이를 다시 반영합니다.
				var fs = PL.Friction.Static;
				var contactVel = Vector3.Cross(A.AngularVelocity * A.Radius, N) * fs;
				contactVel -= Vector3.Scale(contactVel, Context.UpVector);
				contactVel -= Vector3.Scale(contactVel, N); // 노멀 방향 성분, 수직 성분을 제거합니다.
				A.Velocity += contactVel;
				// A.AngularVelocity *= 1 - fs;

				var fv = (float)PL.Damping;
				var horiVec = V1 - V1p;
				var fricVec = -horiVec;
				var angDeltaOrigin = Vector3.Cross(fricVec, N);
				A.AngularVelocity += angDeltaOrigin * fv / A.Radius;

				break;
			}
			}
		}

		Contact? CalcContact(PhysSphere o)
		{
			const float SMALL_NUMBER = Constants.KINDA_SMALL_NUMBER;

			// TODO: 차후 새로운 회전-기반 모델로 변경하는 경우, DeltaFromLastCollision 값을 활용하여 마찰 계수를 결정합니다. 이 때, 아래 식은 t0 ~ t1(구르기 시작), t1~에 대해 서로 다른 마찰 계수로 각각 계산되어야 합니다.

			if (Damping < SMALL_NUMBER)
			{
				throw new Exception("Damping coefficient must be larger than 0.");
			}

			if (Math.Abs(Damping - Damping) > SMALL_NUMBER)
			{
				throw new Exception("Damping coefficient between spheres must be equal");
			}

			Vec3d P1 = Position, P2 = o.Position, V1 = Velocity, V2 = o.Velocity;
			float r1 = Radius, r2 = o.Radius;
			double alpha_inv = 1 / Damping;

			// 오버랩 계산
			var dist0 = ((Vector3)(P1 - P2)).magnitude;
			if ((r1 + r2) - dist0 > SMALL_NUMBER && Vec3d.Dot(V1, V2) < 0)
			{
				Contact ct;
				ct.A = (P1, V1);
				ct.B = (P2, V2);
				ct.At = Vector3.Lerp(P1, P2, r1 / (r1 + r2));
				ct.Time = 0;
				ct.OverlapDepth = (r1 + r2) - dist0;
				return ct;
			}

			Vec3d A, B;
			B = alpha_inv * (V2 - V1);
			A = P2 - P1 + B;

			var a = Vec3d.Dot(B, B);
			var b = Vec3d.Dot(A, B);
			var c = Vec3d.Dot(A, A) - (r2 + r1) * (r2 + r1);

			var discr = b * b - a * c;

			if (discr < SMALL_NUMBER)
			{
				return null;
			}

			var discrSqrt = Math.Sqrt(discr);
			var umin = (b + discrSqrt) / a;
			var umax = (b - discrSqrt) / a;

			double? returnAlphaNonZero(double uu)
			{
				return uu > 0.0f && uu <= 1.0f ? -alpha_inv * Math.Log(uu) : new double?();
			}

			double? t;
			double u;
			if (umin <= 0)
			{
				u = umax;
				t = returnAlphaNonZero(umax);
			}
			else
			{
				u = umin;
				t = returnAlphaNonZero(umin);
			}

			if (t.HasValue)
			{
				Contact contact;
				contact.A.Pos = P1 + V1 * alpha_inv * (1.0 - u);
				contact.A.Vel = V1 * u;
				contact.B.Pos = P2 + V2 * alpha_inv * (1.0 - u);
				contact.B.Vel = V2 * u;
				contact.Time = (float)t.Value;
				contact.At = Vector3.Lerp(contact.A.Pos, contact.B.Pos, r1 / (r1 + r2));
				contact.OverlapDepth = 0;
				return contact;
			}

			return null;
		}

		Contact? CalcContact(PhysStaticPlane o)
		{
			const float SMALL_NUMBER = 1e-7f;

			if (Damping < SMALL_NUMBER)
			{
				throw new Exception("Damping coefficient must be larger than 0.");
			}

			(Vec3d Pp, Vec3d n, Vec3d P0, Vec3d V0) = (o.Position, o.Normal, this.Position, this.Velocity);
			(double alpha, double alpha_inv, double r) = (Damping, 1.0 / Damping, Radius);

			// 속도와 노멀이 항상 마주봐야 합니다.
			if (Vec3d.Dot(n, V0) > 0)
			{
				n = -n;
			}

			// 공이 이미 평면과 겹쳐 있는 경우를 처리합니다.
			// 또한, 평면으로 다가가는 경우만 오버랩 처리합니다.
			var Proj = Vector3.Project(Pp - P0, n);
			var contactDist = Proj.magnitude;
			if (r - contactDist > SMALL_NUMBER && Vector3.Dot(Proj, V0) > 0)
			{
				Contact contact;
				contact.A = (P0, V0);
				contact.B = (Pp, Vector3.zero);
				contact.Time = 0;
				contact.At = P0 - n * contactDist;
				contact.OverlapDepth = (float)(r - contactDist);
				return contact;
			}

			var aiV0 = alpha_inv * V0;
			var upper = Vec3d.Dot(n, (P0 + aiV0 - Pp)) - r;
			var lower = Vec3d.Dot(n, aiV0);

			if (Math.Abs(lower) > SMALL_NUMBER)
			{
				var u = upper / lower;
				var t = -alpha_inv * Math.Log(u);

				if (t > 0)
				{
					var Psph = P0 + aiV0 * (1 - u);
					Contact contact;
					contact.A = (Psph, V0 * u);
					contact.B = (Pp, Vector3.zero);
					contact.At = Psph - r * n;
					contact.Time = (float)t;
					contact.OverlapDepth = 0;

					return contact;
				}
			}

			return null;
		}
	}

	internal class PhysStaticPlane : PhysObject
	{
		public override PhysType Type => PhysType.StaticPlane;
		public Vector3 Normal
		{
			get => _normal;
			set => _normal = value.normalized;
		}

		Vector3 _normal = Vector3.forward;

		public override Contact? CalcMinContactTime(PhysObject other)
		{
			switch (other.Type)
			{
			case PhysType.Sphere:
				var ctopt = other.CalcMinContactTime(this);
				if (ctopt.HasValue)
				{
					var ct = ctopt.Value;
					var tmp = ct.B;
					ct.B = ct.A;
					ct.A = tmp;
					ctopt = ct;
				}

				return ctopt;
			case PhysType.StaticPlane:
				return null;
			default:
				throw new ArgumentOutOfRangeException();
			}
		}

		protected override void AdvanceMovementImpl(float delta)
		{
			// DO NOTHING
		}

		public override void ApplyCollisionImpl(PhysObject other)
		{
			// Only if target is sphere ...
			if (other.Type == PhysType.Sphere)
			{
				((PhysSphere)other).ApplyCollisionImpl(this);
			}
		}
	}

	public struct Vec3d
	{
		double x, y, z;

		public override string ToString()
		{
			return ((Vector3)this).ToString();
		}

		public Vec3d(double x = 0.0, double y = 0.0, double z = 0.0)
		{
			(this.x, this.y, this.z) = (x, y, z);
		}

		public static implicit operator Vector3(Vec3d i)
		{
			return new Vector3((float)i.x, (float)i.y, (float)i.z);
		}

		public static implicit operator Vec3d(Vector3 i)
		{
			Vec3d o;
			(o.x, o.y, o.z) = (i.x, i.y, i.z);
			return o;
		}

		public static Vec3d operator +(Vec3d l, Vec3d r)
		{
			Vec3d o;
			(o.x, o.y, o.z) = (l.x + r.x, l.y + r.y, l.z + r.z);
			return o;
		}

		public static Vec3d operator -(Vec3d l, Vec3d r)
		{
			return l + (-r);
		}

		public static Vec3d operator -(Vec3d l)
		{
			Vec3d o;
			(o.x, o.y, o.z) = (-l.x, -l.y, -l.z);
			return o;
		}

		public static Vec3d operator *(Vec3d a, double b)
		{
			return new Vec3d(a.x * b, a.y * b, a.z * b);
		}

		public static Vec3d operator *(double b, Vec3d a)
		{
			return a * b;
		}

		public static double operator |(Vec3d l, Vec3d r)
		{
			return l.x * r.x + l.y * r.y + l.z * r.z;
		}

		public static double Dot(Vec3d l, Vec3d r)
		{
			return l | r;
		}
	}

}

