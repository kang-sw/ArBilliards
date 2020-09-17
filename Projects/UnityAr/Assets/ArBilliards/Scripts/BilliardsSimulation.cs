using System;
using System.Collections;
using System.Collections.Generic;
using Boo.Lang.Runtime;
using JetBrains.Annotations;
using UnityEditor.Compilation;
using UnityEditor.UIElements;
using UnityEngine;

namespace Billiards.Simulation
{
	public struct Recognitions
	{
		public Vector3 Red1;
		public Vector3 Red2;
		public Vector3 Orange;
		public Vector3 White;
	}

	// TODO: 구 - 구 충돌, 구 - 정적 평면 ==> 직선 - 정적 평면 충돌
	// TODO: 반발 계수 공식 
	// TODO: 정해진 time step에 대해 모든 충돌체 iterate하여 가장 
	public class Simulator : PhysContext
	{
		#region Simulation Properties

		public (float Min, float Max) ImpactRange;
		public float BallDampingCoeff = 0.02f;
		public float RestitutionCoeff = 0.87f;

		#endregion

		public enum BallIndex
		{
			Red1,
			Red2,
			Orange,
			White
		};

		public class SimulationResult
		{
			private bool bHasPoint = false;
			public List<(BallIndex Ball, List<Vector3> Paths)> Visits = new List<(BallIndex Ball, List<Vector3> Paths)>();
			public Vector3 InitialImpactPoint = Vector3.zero; // 충격이 시작되는 점입니다.
			public Vector3 InitialImpactDirection = Vector3.zero; // 충격량 + 방향 벡터입니다.
		}

		public List<SimulationResult> SolveSimulation(Recognitions result)
		{
			return null;
		}

		public class SimulationTriggerParam : ICloneable
		{
			public (Vector3 Start, Vector3 End) ImpactPath = (Vector3.zero, Vector3.zero);
			public float Mass = 1.0f;
			public float SimDuration = 1.0f;

			public object Clone()
			{
				var ret = new SimulationTriggerParam();
				ret.ImpactPath = ImpactPath;
				ret.Mass = Mass;
				ret.SimDuration = SimDuration;
				return ret;
			}
		}

		public SimulationResult Simulate(Recognitions result, SimulationTriggerParam triggerParam)
		{

			return null;
		}

	}

	public class PhysContext
	{

	}

	enum PhysType
	{
		Sphere,
		StaticPlane
	}

	internal abstract class PhysObject
	{
		public abstract PhysType Type { get; }
		public float Mass = 0;
		public Vector3 Position = new Vector3();
		public Vector3 Velocity = new Vector3();
		public float DampingCoeff = 0.001f;
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
			var eat = Mathf.Exp(-DampingCoeff * delta);
			Position += Velocity * (1.0f / DampingCoeff) * (1.0f - eat);
			Velocity *= eat;
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

		public override void ApplyCollision(PhysObject other)
		{
			throw new NotImplementedException();
		}

		Contact? CalcContact(PhysSphere o)
		{
			const float SMALL_NUMBER = 1e-7f;

			if (DampingCoeff < SMALL_NUMBER)
			{
				throw new AssertionFailedException("Damping coefficient must be larger than 0.");
			}

			if (Math.Abs(DampingCoeff - DampingCoeff) < SMALL_NUMBER)
			{
				throw new AssertionFailedException("Damping coefficient between spheres must be equal");
			}

			Vector3 P1 = Position, P2 = o.Position, V1 = Velocity, V2 = o.Velocity;
			float r1 = Radius, r2 = o.Radius;
			float alpha_inv = 1 / DampingCoeff;
			Vector3 A, B;

			B = V2 - V1;
			A = P2 - P1 + alpha_inv * B;

			float a = Vector3.Dot(B, B);
			float b = Vector3.Dot(A, B);
			float c = Vector3.Dot(A, A) - (r2 + r1) * (r2 + r1);

			float discr = b * b - a * c;

			if (discr < SMALL_NUMBER)
			{
				return null;
			}

			discr = Mathf.Sqrt(discr);
			float umin = (b - discr) / a;
			float umax = (b + discr) / a;


			float? returnAlphaNonZero(float u)
			{

				return u > 0.0f && u <= 1.0f ? -alpha_inv * Mathf.Log(u) : new float?();
			}

			float? t;
			float u;
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
				contact.A.Pos = P1 + V1 * alpha_inv * (1.0f - u);
				contact.A.Vel = V1 * u;
				contact.B.Pos = P2 + V2 * alpha_inv * (1.0f - u);
				contact.B.Vel = V2 * u;
				contact.Time = t.Value;
				return contact;
			}

			return null;
		}

		Contact? CalcContact(PhysStaticPlane other)
		{
			return null;
		}
	}

	internal class PhysStaticPlane : PhysObject
	{
		public override PhysType Type => PhysType.StaticPlane;

		public override Contact? CalcMinContactTime(PhysObject other)
		{
			switch (other.Type)
			{
			case PhysType.Sphere:
				return other.CalcMinContactTime(this);
			case PhysType.StaticPlane:
				return null;
			default:
				throw new ArgumentOutOfRangeException();
			}
		}

		public override void AdvanceMovement(float delta)
		{
			throw new NotImplementedException();
		}

		public override void ApplyCollision(PhysObject other)
		{
			throw new NotImplementedException();
		}
	}

}
