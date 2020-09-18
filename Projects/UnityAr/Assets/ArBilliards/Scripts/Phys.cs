using System;
using System.CodeDom.Compiler;
using System.Collections;
using System.Collections.Generic;
using Boo.Lang.Runtime;
using JetBrains.Annotations;
using UnityEditor.Compilation;
using UnityEditor.UIElements;
using UnityEngine;
using Object = System.Object;

namespace ArBilliards.Phys
{
	public static class Constants
	{
		public const float KINDA_SMALL_NUMBER = 1e-6f;
	}

	public class PhysContext
	{
		public float OverlapPushVelocity = 1f;
		public float OverlapSteps = 0.01f; // Overlap된 물체가 하나라도 있을 경우 resolve될 때까지 스탭을 아래의 값으로 고정합니다.

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

			// 남은 시간이 0이 될 때까지 반복합니다.
			while (deltaTime > 0)
			{
				float minContactTime = deltaTime;
				(int A, int B)? minContactIdx = null;
				PhysObject.Contact? minContact = null;

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
							if (ct.A.Vel.sqrMagnitude + ct.B.Vel.sqrMagnitude == 0 || overlaps.Contains(B))
							{
								continue;
							}
							else
							{
								overlaps.Add(B);
							}
						}

						if (ct.Time < minContactTime)
						{
							minContactIdx = (idxA, idxB);
							minContactTime = contact.Value.Time;
							minContact = contact;
						}
					}
				}

				// 최소 델타 시간만큼 모든 오브젝트를 전진시킵니다.
				foreach (var elem in Enumerable)
				{
					elem.AdvanceMovement(minContactTime);
				}

				totalTime += minContactTime;

				if (minContactIdx.HasValue)
				{
					// 충돌을 기록합니다.
					var idx = minContactIdx.Value;
					var (A, B) = (this[idx.A], this[idx.B]);

					ContactInfo contact;
					contact.A = (idx.A, A.Position, A.Velocity);
					contact.B = (idx.B, B.Position, B.Velocity);
					contact.Time = totalTime;
					contact.At = minContact.Value.At;

					result.Add(contact);

					// 충돌을 계산합니다.
					A.ApplyCollision(B);
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
		public float Mass = 1.0f;
		public Vector3 Position = new Vector3();
		public Vector3 Velocity = new Vector3();
		public double DampingCoeff = 0.001;
		public float RestitutionCoeff = 1.0f;

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
		public virtual void AdvanceMovement(float delta)
		{
			var eat = Math.Exp(-DampingCoeff * delta);
			Vec3d V = Velocity;
			Vec3d P = Position;

			Velocity = V * eat;
			Position = P + V * (1.0 / DampingCoeff) * (1.0f - eat);
		}

		/// <summary>
		/// 두 물체 사이의 충돌 연산을 수행하고 물리 상태를 변화시킵니다.
		/// 위치는 고려하지 않으며, 속도와 질량 등 여러 물리 계수를 고려해 현재 속도를 변화시킵니다.
		/// </summary>
		/// <param name="other">연산을 적용할 대상 오브젝트입니다.</param>
		/// <returns></returns>
		public abstract void ApplyCollision(PhysObject other);

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
	}

	internal class PhysSphere : PhysObject
	{
		public float Radius = 1.0f;

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

		public override void ApplyCollision(PhysObject B)
		{
			// 충돌 각도를 계산하기 위해, 먼저 두 구의 중심선에 대한 벡터 투영을 구합니다.
			// ref: https://sites.google.com/site/3dgameprogram/home/physics-modeling-for-game-programming/-gibongaenyeomdajigi---mulli/-jiljeom-ui-chungdol
			const float SMALL_NUMBER = Constants.KINDA_SMALL_NUMBER;
			var A = this;

			var (e, m1, m2) = (0.5f * (A.RestitutionCoeff + B.RestitutionCoeff), A.Mass, B.Mass);
			var (V1, V2) = (A.Velocity, B.Velocity);

			switch (B.Type)
			{
			case PhysType.Sphere:
			{
				var center = (B.Position - A.Position).normalized;

				var V1p = Vector3.Project(V1, center);
				var V2p = Vector3.Project(V2, -center);

				var nV1p = ((m1 - e * m2) * V1p + (1 + e) * m2 * V2p) / (m1 + m2);
				var nV2p = ((m2 - e * m1) * V2p + (1 + e) * m1 * V1p) / (m1 + m2);

				var nV1 = V1 + (nV1p - V1p);
				var nV2 = V2 + (nV2p - V2p);

				A.Velocity = nV1;
				B.Velocity = nV2;

				break;
			}
			case PhysType.StaticPlane:
			{
				// Plane의 
				var PL = (PhysStaticPlane)B;
				var N = PL.Normal;

				var V1p = Vector3.Project(V1, N);
				var nV1p = -e * V1p;

				var nV1 = V1 + (nV1p - V1p);
				A.Velocity = nV1;

				break;
			}
			}
		}

		Contact? CalcContact(PhysSphere o)
		{
			const float SMALL_NUMBER = Constants.KINDA_SMALL_NUMBER;

			if (DampingCoeff < SMALL_NUMBER)
			{
				throw new AssertionFailedException("Damping coefficient must be larger than 0.");
			}

			if (Math.Abs(DampingCoeff - DampingCoeff) > SMALL_NUMBER)
			{
				throw new AssertionFailedException("Damping coefficient between spheres must be equal");
			}

			Vec3d P1 = Position, P2 = o.Position, V1 = Velocity, V2 = o.Velocity;
			float r1 = Radius, r2 = o.Radius;
			double alpha_inv = 1 / DampingCoeff;
			Vec3d A, B;

			// 오버랩 계산
			var dist0 = ((Vector3)(P1 - P2)).magnitude;
			if ((r1 + r2) - dist0 > 0 && Vec3d.Dot(V1, V2) < 0)
			{
				Contact ct;
				ct.A = (P1, V1);
				ct.B = (P2, V2);
				ct.At = Vector3.Lerp(P1, P2, r1 / (r1 + r2));
				ct.Time = 0;
				ct.OverlapDepth = (r1 + r2) - dist0;
				return ct;
			}

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

			if (DampingCoeff < SMALL_NUMBER)
			{
				throw new AssertionFailedException("Damping coefficient must be larger than 0.");
			}

			(Vec3d Pp, Vec3d n, Vec3d P0, Vec3d V0) = (o.Position, o.Normal, this.Position, this.Velocity);
			(double alpha, double alpha_inv, double r) = (DampingCoeff, 1.0 / DampingCoeff, Radius);

			// 속도와 노멀이 항상 마주봐야 합니다.
			if (Vec3d.Dot(n, V0) > 0)
			{
				n = -n;
			}

			// 공이 이미 평면과 겹쳐 있는 경우를 처리합니다.
			// 또한, 평면으로 다가가는 경우만 오버랩 처리합니다.
			var Proj = Vector3.Project(Pp - P0, n);
			var contactDist = Proj.magnitude;
			if (r - contactDist > 0 && Vector3.Dot(Proj, V0) > 0)
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

		public override void AdvanceMovement(float delta)
		{
			// DO NOTHING
		}

		public override void ApplyCollision(PhysObject other)
		{
			// Only if target is sphere ...
			if (other.Type == PhysType.Sphere)
			{
				((PhysSphere)other).ApplyCollision(this);
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
