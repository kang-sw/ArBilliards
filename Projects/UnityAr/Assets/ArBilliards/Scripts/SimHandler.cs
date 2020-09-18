﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using ArBilliards.Phys;
using Boo.Lang.Runtime;
using UnityEngine;

public enum BilliardsBall
{
	Red1, Red2, Orange, White
}

/// <summary>
/// 경로 시뮬레이션에 관련된 작업을 수행하는 클래스입니다.
/// 다수의 각도로 경로 시뮬레이션을 수행해 캐시합니다.
///
/// 사용자가 어떤 공을 칠 차례인지 지정하면 해당 공을 기준으로 시뮬레이션을 실행합니다.
///
/// 사용자의 카메라 각도에 따라 자동으로 해당 방향의 가장 vote가 높은 시뮬레이션 경로를 강조합니다.
///
/// 시뮬레이션은 비동기적으로 수행됩니다.
/// </summary>
public class SimHandler : MonoBehaviour
{
	#region Public Properties

	[Header("General")]
	public Transform RenderingAnchor; // 테이블의 원점이 될 루트입니다.

	[Header("Dimensions")]
	public float BallRadius = 0.07f / Mathf.PI;
	public float TableWidth = 0.96f;
	public float TableHeight = 0.51f;

	[Header("Coefficients")]
	public float BallRestitution = 0.71f;
	public float BallDamping = 0.33f;
	public float TableRestitution = 0.53f;

	[Header("Optimizations")]
	public int NumRotationDivider = 360;
	public float[] Velocities = new[] { 0.4f };
	public float SimDuration = 10.0f;

	[Header("GameState")]
	public BilliardsBall PlayerBall;

	#endregion

	private AsyncSimAgent _sim = new AsyncSimAgent();
	private AsyncSimAgent.InitParams _param;

	// Start is called before the first frame update
	void Start()
	{
		AsyncSimAgent.InitParams.Rule4Balls(ref _param);
	}

	// Update is called once per frame
	void Update()
	{
	}



	void UpdateParam(ref AsyncSimAgent.InitParams p)
	{
		p.SimDuration = SimDuration;
		p.Ball = (BallRestitution, BallDamping, BallRadius);
		p.PlayerBall = PlayerBall;
		p.Table = (TableRestitution, TableWidth, TableHeight);
		p.NumCandidates = NumRotationDivider;
	}
}

/// <summary>
/// 비동기 시뮬레이션 에이전트입니다.
///
/// 시뮬레이션은 ZX 평면으로 고정됩니다.
/// </summary>
public class AsyncSimAgent
{
	public struct InitParams
	{
		// DATA
		public Vector3 Red1, Red2, Orange, White; // 테이블 원점 0, 0에 대한 당구공 4개의 좌표
		public (float Restitution, float Width, float Height) Table; // 테이블 속성
		public (float Restitution, float Damping, float Radius) Ball; // 공 속성

		// RULES 
		public BilliardsBall PlayerBall; // 플레이어가 칠 공입니다.
		public bool bAvoidPlayerBallHit; // 4구 룰에서, 다른 플레이어의 공을 치면 실점 룰 적용
		public bool bOpponentBallAsScore; // 상대 공을 치는것을 점수로 칠건지 결정합니다. 3구 룰.
		public int NumCushionHits; // 마지막 공 타격 전까지의 최소 쿠션 히트 수입니다.

		// OPTIONS
		public float InitSpeed; // 타격하는 공의 최초 속도입니다.

		// OPTIMIZATION
		public int NumCandidates; // 360도 범위에서 몇 개의 후보를 선택할지 결정합니다. 후보는 uniform하게 선정됩니다.
		public float SimDuration; // 총 시뮬레이션 길이입니다.

		public static void Rule3Balls(ref InitParams p)
		{
			p.bAvoidPlayerBallHit = false;
			p.bOpponentBallAsScore = true;
			p.NumCushionHits = 3;
		}

		public static void Rule4Balls(ref InitParams p)
		{
			p.bAvoidPlayerBallHit = true;
			p.bOpponentBallAsScore = false;
			p.NumCushionHits = 0;
		}
	}


	public sealed class SimResult
	{
		public class Candidate
		{
			public BallPath[] Balls; // 4개 항목
			public float Votes; // 해당 결과의 확실성입니다.
			public Vector3 Direction; // 타격 방향 (노멀)

			public Candidate()
			{
				Balls = new BallPath[4];
				for (int i = 0; i < Balls.Length; ++i)
					Balls[i] = new BallPath();

				Votes = 0f;
				Direction = Vector3.zero;
			}
		}

		// 시작 시 부여한 조건입니다.
		public InitParams Options = new InitParams();

		// vote 순서대로 정렬된 가능성 높은 시뮬레이션 결과 목록
		public List<Candidate> Candidates = new List<Candidate>();

		public SimResult Clone()
		{
			return (SimResult)MemberwiseClone();
		}
	}

	public class BallPath
	{
		public struct Node
		{
			public BilliardsBall? Other; // 충돌변인. null이면 벽에 충돌
			public float Time; // 충돌 시점
			public Vector3 Position;
			public Vector3 Velocity; // 해당 위치 통과 시점 속도
		}

		public BilliardsBall Index;
		public readonly List<Node> Nodes = new List<Node>();
	}

	#region Public Props

	/// <summary>
	/// 시뮬레이션 결과가 준비된 경우 true 반환합니다.
	/// </summary>
	public bool IsRunning { get; private set; }

	public SimResult SimulationResult => IsRunning ? _simRes : null;

	#endregion

	/// <summary>
	/// 비동기 시뮬레이션을 트리거합니다.
	/// 
	/// </summary>
	/// <param name="param"></param>
	public async void InitAsync(InitParams param)
	{
		if (IsRunning)
		{
			throw new AssertionFailedException("Async process still running!");
		}

		IsRunning = true;
		_p = param;
		await new Task(internalAsyncJob);
		IsRunning = false;
	}

	public void InitSync(InitParams param)
	{
		if (IsRunning)
		{
			return;
		}

		IsRunning = true;
		_p = param;
		internalAsyncJob();
		IsRunning = false;
	}

	#region Internal props to handle async process

	private PhysContext _sim;
	private SimResult _simRes;
	private InitParams _p;

	private (float x, float y)[] _wallPositions;
	private Vector3[] _ballPositions;
	private readonly int[] _ballIndexes = new int[4];
	private readonly PhysObject[] _ballRefs = new PhysObject[4];

	#endregion


	void internalAsyncJob()
	{
		// 초기 오브젝트를 스폰합니다.
		// 8개의 격벽(Inner, Outer), 4개의 공 
		initSim();

		var contacts = new List<PhysContext.ContactInfo>();
		var res = _simRes = new SimResult();
		res.Options = _p;

		for (int iter = 0, maxIter = _p.NumCandidates; iter < maxIter; ++iter)
		{
			resetBallState();

			// -- 시뮬레이션 셋업
			var cand = new SimResult.Candidate();
			var balls = cand.Balls;
			res.Candidates.Add(cand);

			// -- 초기 속력 및 방향 지정
			float angle = (2f * Mathf.PI / maxIter) * iter;
			var dir = Quaternion.Euler(0, angle, 0) * Vector3.forward;
			cand.Direction = dir;

			var initVelocity = _p.InitSpeed * dir;
			_ballRefs[(int)_p.PlayerBall].Velocity = initVelocity;

			// -- 공 초기 위치 노드 셋업
			for (int i = 0; i < 4; ++i)
			{
				var r = _ballRefs[i];

				BallPath.Node n;
				n.Position = r.Position;
				n.Velocity = r.Velocity;
				n.Time = 0f;
				n.Other = null;

				balls[i].Nodes.Add(n);
				balls[i].Index = (BilliardsBall)i;
			}

			// -- 시뮬레이션 트리거 후 충돌 수집
			_sim.StepSimulation(_p.SimDuration, contacts);

			// 충돌 위치 목록 순회하며 공 경로 분석
			foreach (var contact in contacts)
			{
				var ct = contact;
				var A = getBallId(ct.A.Idx);
				var B = getBallId(ct.B.Idx);

				for (int swap = 0; swap < 2; ++swap)
				{
					if (A.HasValue)
					{
						var ballIndex = (int)A.Value;

						BallPath.Node n;
						n.Other = B;
						n.Position = ct.A.Pos;
						n.Velocity = ct.A.Vel;
						n.Time = ct.Time;

						balls[ballIndex].Nodes.Add(n);
					}

					var tmp = ct.A;
					ct.A = ct.B;
					ct.B = ct.A;
				}
			}

			// -- 플레이어 공 분석하여 득점 여부 계산
			// TODO:
		}

		// 끝
	}

	private void initSim()
	{
		_sim = new PhysContext();

		var (rst, w, h) = (_p.Table.Restitution, _p.Table.Width, _p.Table.Height);
		_wallPositions = new[]
		{
			(w, 0f), (-w, 0f), (w * 1.05f, 0f), (-w * 1.05f, 0f),
			(0f, h), (0f, -h), (0f, h * 1.05f), (0f, h * -1.05f)
		};

		var spn = new PhysStaticPlane();
		spn.RestitutionCoeff = rst;

		foreach (var pos in _wallPositions)
		{
			// 위치, 노멀 설정은 아래에서 ...
			spn.Position = new Vector3(pos.x, 0, pos.y);
			spn.Normal = new Vector3(-pos.x, 0, -pos.y);
			_sim.Spawn(spn);
		}

		// 공 스폰 및 초기 위치 목록 작성
		_p.Red1.y = _p.Red2.y = _p.Orange.y = _p.White.y = 0;
		_ballPositions = new[] { _p.Red1, _p.Red2, _p.Orange, _p.White };

		var ball = new PhysSphere();
		ball.Radius = _p.Ball.Radius;
		ball.RestitutionCoeff = _p.Ball.Restitution;
		ball.DampingCoeff = _p.Ball.Damping;

		// 인덱스 맵 생성
		for (int index = 0; index < 4; index++)
		{
			var ballIndex = _sim.Spawn(ball);
			var ballRef = _sim[ballIndex];
			_ballRefs[index] = ballRef;
		}
	}

	BilliardsBall? getBallId(int ballIndex)
	{
		for (int index = 0; index < 4; ++index)
		{
			if (ballIndex == _ballIndexes[index])
			{
				return (BilliardsBall)index;
			}
		}

		return null;
	}

	void resetBallState()
	{
		_sim.ResetOverlapStates();

		for (int index = 0; index < 4; index++)
		{
			var ballRef = _ballRefs[index];
			ballRef.Position = _ballPositions[index];
			ballRef.Velocity = Vector3.zero;
		}
	}
}
