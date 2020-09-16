using System;
using System.Collections;
using System.Collections.Generic;
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
}
