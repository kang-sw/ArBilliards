using System;
using System.Collections;
using System.Collections.Generic;
using Boo.Lang.Runtime;
using JetBrains.Annotations;
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
	public class Context
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

		public List<SimulationResult> SolveSimulation(Matrix4x4 tableTransformInv, Recognitions result)
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

		public SimulationResult Simulate(Matrix4x4 tableTransform, Recognitions result, SimulationTriggerParam triggerParam)
		{

			return null;
		}

	}

	enum PhysType
	{
		Sphere,
		StaticPlane
	}

	internal abstract class PhysObject
	{
		public PhysType Type { get; protected set; }
		public float Mass = 0;
		public Vector3 Position = new Vector3();
		public Vector3 Velocity = new Vector3();
		public float DampingCoeff = 0.001f;
		public float RestitutionCoeff = 1.0f;

		public abstract float? CalcMinContactTime(PhysObject other);
		public abstract Contact? AdvanceSimulation(PhysObject other, float delta);

		public struct Contact
		{
			public (Vector3 Pos, Vector3 Vel) A, B;
			public float Time;
		}
	}

	internal class PhysSphere : PhysObject
	{
		public float Radius = 1.0f;

		public PhysSphere()
		{
			Type = PhysType.Sphere;
		}

		public override float? CalcMinContactTime(PhysObject other)
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

		public override Contact? AdvanceSimulation(PhysObject other, float delta)
		{
			throw new NotImplementedException();
		}

		public float? CalcContact(PhysSphere o)
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
			float alpha = DampingCoeff;
			Vector3 A, B;

			B = V2 - V1;
			A = P2 - P1 + (1 / alpha) * B;

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
				return u > 0.0f && u <= 1.0f ? Mathf.Log(u) : new float?();
			}

			if (umin <= 0)
			{
				return returnAlphaNonZero(umax);
			}
			else
			{
				return returnAlphaNonZero(umin);
			} 
		}

		public float? CalcContact(PhysStaticPlane other)
		{
			return null;
		}
	}

	internal class PhysStaticPlane : PhysObject
	{
		public PhysStaticPlane()
		{
			Type = PhysType.StaticPlane;
		}

		public override float? CalcMinContactTime(PhysObject other)
		{
			switch (other.Type)
			{
			case PhysType.Sphere:
				return ((PhysSphere)other).CalcContact(this);
			case PhysType.StaticPlane:
				return null;
			default:
				throw new ArgumentOutOfRangeException();
			} 
		}

		public override Contact? AdvanceSimulation(PhysObject other, float delta)
		{
			throw new NotImplementedException();
		}
	}

}
