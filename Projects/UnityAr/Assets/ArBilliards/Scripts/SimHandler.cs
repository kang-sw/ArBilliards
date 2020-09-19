﻿using System;
using ArBilliards.Phys;
using Boo.Lang.Runtime;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Messaging;
using System.Threading.Tasks;
using UnityEngine;
using Random = System.Random;

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
	public Transform TableAnchor; // 렌더링 루트입니다.
	public Transform LookTransform; // 전방 방향 트랜스폼
	public float SimInterval = 0.5f; // 시뮬레이션 간격입니다.

	[Header("Dimensions")]
	public float BallRadius = 0.07f / Mathf.PI;
	public float TableWidth = 0.96f;
	public float TableHeight = 0.51f;

	[Header("Coefficients")]
	public float BallRestitution = 0.71f;
	public float BallDamping = 0.33f;
	public float TableRestitution = 0.53f;
	public float TableFriction = 0.22f;

	[Header("Optimizations")]
	public int NumRotationDivider = 360;
	public float SimDuration = 10.0f;
	public float SpeedMin = 0.4f;
	public float SpeedMax = 1.4f;

	[Header("GameState")]
	public BilliardsBall PlayerBall = BilliardsBall.Orange;

	[Header("Visualizer Instances")]
	public Color[] BallVisualizeColors = new Color[4];
	public LineRenderer CandidateRenderer;
	public LineRenderer[] ColorRenderers = new LineRenderer[4];

	[Header("Visualizer Templates")]
	public GameObject PathFollowMarkerTemplate;
	public GameObject CandidateRenderingTemplate;
	public GameObject CollisionMarkerTemplate;

	#endregion

	#region Public States

	public (Vector3 Red1, Vector3 Red2, Vector3 Orange, Vector3 White)? PendingBallPositions { get; set; }

	#endregion

	#region Main Loop

	private GameObject _instancedRoot;
	private AsyncSimAgent.SimResult _latestResult = null;
	private bool _bLineDirty = false;

	// Start is called before the first frame update
	void Start()
	{
		_instancedRoot = new GameObject("InstantiationRoot");
		_instancedRoot.transform.parent = TableAnchor;

		Start_AsyncSimulation();
	}

	// Update is called once per frame
	void Update()
	{
		Update_AsyncSimulation();

		if (_bLineDirty && LookTransform && _latestResult.Candidates.Count > 0)
		{
			var fwd = LookTransform.forward;
			var r = _latestResult;

			// 현재 카메라 방향에 따라 가장 적합한 candidate를 찾습니다.
			fwd = TableAnchor.worldToLocalMatrix.MultiplyVector(fwd);
			AsyncSimAgent.SimResult.Candidate nearlest = r.Candidates[0];

			foreach (var elem in r.Candidates)
			{
				if (Vector3.Angle(nearlest.InitVelocity, fwd) > Vector3.Angle(elem.InitVelocity, fwd))
					nearlest = elem;
			}

			// Render candidate markers


			initMarkerPool();
			for (int index = 0; index < 4; ++index)
			{
				renderBallPath(ColorRenderers[index], nearlest.Balls[index]);
			}
			trimUnusedMarkers();

		}
	}

	void updateParam(ref AsyncSimAgent.InitParams p)
	{
		p.SimDuration = SimDuration;
		p.Ball = (BallRestitution, BallDamping, BallRadius);
		p.PlayerBall = PlayerBall;
		p.Table = (TableRestitution, TableWidth, TableHeight, TableFriction);
		p.NumCandidates = NumRotationDivider;
	}

	#endregion

	#region Simulations

	private AsyncSimAgent _sim = new AsyncSimAgent();
	private float _intervalTimer = 0f;
	private bool _bParallelProcessRunning;

	void Start_AsyncSimulation()
	{

	}

	void Update_AsyncSimulation()
	{
		if (_sim.SimulationResult != null)
		{
			_latestResult = _sim.SimulationResult;
			_bLineDirty = true;
		}

		_intervalTimer += Time.deltaTime;
		if (_intervalTimer > SimInterval && !_sim.IsRunning && PendingBallPositions.HasValue)
		{
			// 비동기 시뮬레이션 트리거
			var p = new AsyncSimAgent.InitParams();
			updateParam(ref p);
			AsyncSimAgent.InitParams.Rule4Balls(ref p);

			(p.Red1, p.Red2, p.Orange, p.White) = PendingBallPositions.Value;
			PendingBallPositions = null;

			p.InitSpeedRange = (SpeedMin, SpeedMax);
			_sim.InitAsync(p);
			_intervalTimer = 0f;
		}
	}

	#endregion

	#region Visualizers

	private List<CollisionMarkerManipulator> _collisionMarkerPool = new List<CollisionMarkerManipulator>();
	private int _numActiveCollisionMarkers;
	private int _cachedNumActiveCollisionMarkers;

	private List<LineRenderer> _candidateMarkerPool = new List<LineRenderer>();
	private int _numActiveCandidateMarkers;

	void renderBallPath(LineRenderer target, AsyncSimAgent.BallPath path)
	{
		if (target == null)
			return;

		// -- 셋업
		// 라인 렌더러를 설정합니다.
		var nodes = path.Nodes;
		target.positionCount = nodes.Count;
		target.useWorldSpace = false;
		target.alignment = LineAlignment.View;

		target.material.color = target.startColor;

		// 해당 볼 색상
		var color = BallVisualizeColors[(int)path.Index];

		// 콜리전 마커 예약
		reserveCollisionMarkers(nodes.Count);

		// -- 그리기 루프
		for (int index = 0; index < nodes.Count; index++)
		{
			// 라인을 그립니다.
			target.SetPosition(index, nodes[index].Position);

			// 각 조인트마다 충돌 마커를 스폰합니다.
			if (index < nodes.Count - 1)
			{
				var marker = spawnCollisionMarker();
				marker.ParticleColor = color;
				color.a = 0.765f;
				marker.MeshColor = color;
				marker.transform.localPosition = nodes[index].Position;
			}
		}

	}

	void initMarkerPool()
	{
		_cachedNumActiveCollisionMarkers = 0;
	}

	void trimUnusedMarkers()
	{
		// 초과분을 비활성화
		for (int i = _cachedNumActiveCollisionMarkers; i < _numActiveCollisionMarkers; i++)
		{
			_collisionMarkerPool[i].Active = false;
		}

		_numActiveCollisionMarkers = _cachedNumActiveCollisionMarkers;
	}

	void reserveCollisionMarkers(int size)
	{
		var desiredSize = size + _cachedNumActiveCollisionMarkers;
		while (_collisionMarkerPool.Count < desiredSize)
		{
			var obj = Instantiate(CollisionMarkerTemplate, TableAnchor);
			obj.SetActive(false);
			obj.transform.localScale = Vector3.one * BallRadius * 2f;
			_collisionMarkerPool.Add(obj.GetComponent<CollisionMarkerManipulator>());
		}
	}

	CollisionMarkerManipulator spawnCollisionMarker()
	{
		var ret = _collisionMarkerPool[_cachedNumActiveCollisionMarkers];
		if (!ret.Active)
			ret.Active = true;

		++_cachedNumActiveCollisionMarkers;
		return ret;
	}


	void debugRenderBallPath(AsyncSimAgent.SimResult.Candidate cand)
	{
		var mtrx = TableAnchor.localToWorldMatrix;

		foreach (var ball in cand.Balls)
		{
			for (int index = 0; index < ball.Nodes.Count - 1; index++)
			{
				var cur = ball.Nodes[index];
				var nxt = ball.Nodes[index + 1];

				var beginPt = mtrx.MultiplyPoint(cur.Position);
				var endPt = mtrx.MultiplyPoint(nxt.Position);

				Debug.DrawLine(beginPt, endPt, BallVisualizeColors[(int)ball.Index]);
			}
		}
	}


	#endregion
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
		public (float Restitution, float Width, float Height, float Friction) Table; // 테이블 속성
		public (float Restitution, float Damping, float Radius) Ball; // 공 속성

		// RULES 
		public BilliardsBall PlayerBall; // 플레이어가 칠 공입니다.
		public bool bAvoidPlayerBallHit; // 4구 룰에서, 다른 플레이어의 공을 치면 실점 룰 적용
		public bool bOpponentBallAsScore; // 상대 공을 치는것을 점수로 칠건지 결정합니다. 3구 룰.
		public int NumCushionHits; // 마지막 공 타격 전까지의 최소 쿠션 히트 수입니다.

		// OPTIONS
		public (float Min, float Max) InitSpeedRange; // 최초 타구 시 속도입니다.

		// OPTIMIZATION
		public int NumCandidates; // 360도 범위에서 몇 개의 후보를 선택할지 결정합니다. 후보는 uniform하게 선정됩니다.
		public (int Begin, int End)? PartialRange; // 다수의 Agent 인스턴스를 생성할 때 유용합니다.
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
			public Vector3 InitVelocity; // 타격 방향 (노멀)

			public Candidate()
			{
				Balls = new BallPath[4];
				for (int i = 0; i < Balls.Length; ++i)
					Balls[i] = new BallPath();

				Votes = 0f;
				InitVelocity = Vector3.zero;
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

	public SimResult SimulationResult => !IsRunning ? _simRes : null;

	#endregion

	/// <summary>
	/// 비동기 시뮬레이션을 트리거합니다.
	/// 
	/// </summary>
	/// <param name="param"></param>
	public async Task<SimResult> InitAsync(InitParams param)
	{
		if (IsRunning)
		{
			throw new AssertionFailedException("Async process still running!");
		}

		IsRunning = true;
		_p = param;
		var task = new Task<SimResult>(internalExec);

		task.Start();

		var result = await task;
		IsRunning = false;

		return result;
	}

	public SimResult InitSync(InitParams param)
	{
		if (IsRunning)
		{
			throw new AssertionFailedException("Async process still running!");
		}

		IsRunning = true;
		_p = param;

		var result = internalExec();
		IsRunning = false;

		return result;
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


	SimResult internalExec()
	{
		// 초기 오브젝트를 스폰합니다.
		// 8개의 격벽(Inner, Outer), 4개의 공 
		initSim();

		var contacts = new List<PhysContext.ContactInfo>();
		var res = _simRes = new SimResult();
		res.Options = _p;

		int iter, maxIter;
		if (_p.PartialRange.HasValue)
			(iter, maxIter) = _p.PartialRange.Value;
		else
			(iter, maxIter) = (0, _p.NumCandidates);

		SimResult.Candidate cand = null;
		var rand = new Random();

		for (; iter < maxIter; ++iter)
		{
			resetBallState();

			// -- 시뮬레이션 셋업
			cand = cand ?? new SimResult.Candidate(); // candidate를 찾는 데 실패한 경우 메모리를 재활용하기 위함입니다.
			var balls = cand.Balls;

			// -- 초기 속력 및 방향 지정
			float angle = (360f / maxIter) * iter;
			var dir = Quaternion.Euler(0, angle, 0) * Vector3.forward;

			var (spdMin, spdMax) = _p.InitSpeedRange;
			var spdCoeff = (float)rand.NextDouble();

			var initVelocity = Mathf.Lerp(spdMin, spdMax, spdCoeff) * dir;
			cand.InitVelocity = dir;
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

					{
						var tmp = ct.A;
						ct.A = ct.B;
						ct.B = tmp;
					}
					{
						var tmp = A;
						A = B;
						B = tmp;
					}

				}
			}

			// -- 공의 최종 위치 추가하기
			for (int i = 0; i < 4; ++i)
			{
				var r = _ballRefs[i];

				BallPath.Node n;
				n.Position = r.Position;
				n.Velocity = r.Velocity;
				n.Time = _p.SimDuration;
				n.Other = null;

				balls[i].Nodes.Add(n);
				balls[i].Index = (BilliardsBall)i;
			}

			// -- 플레이어 공 분석하여 득점 여부 계산
			bool bGotScore = false;
			var hits = new (int bHit, int numLastCushion)[4];
			int numCushionHit = 0;

			foreach (var node in balls[(int)_p.PlayerBall].Nodes)
			{
				if (!node.Other.HasValue) // Other가 비었으면 벽입니다.
				{
					++numCushionHit;
				}
				else
				{
					var index = (int)node.Other.Value;
					hits[index] = (1, numCushionHit);
				}
			}

			// 득점 조건 검사
			var otherPlayer = _p.PlayerBall == BilliardsBall.White ? BilliardsBall.Orange : BilliardsBall.White;
			int maxCushions = Math.Max(hits[0].numLastCushion, hits[1].numLastCushion);
			int hitBalls = hits[0].bHit + hits[1].bHit;

			if (hits[(int)otherPlayer].bHit == 1)
			{
				hitBalls += _p.bOpponentBallAsScore ? 1 : _p.bAvoidPlayerBallHit ? -2 : 0;
			}

			if (_p.bOpponentBallAsScore)
				maxCushions = Math.Max(maxCushions, hits[(int)otherPlayer].numLastCushion);

			if (maxCushions >= _p.NumCushionHits && hitBalls >= 2)
			{
				bGotScore = true;
			}

			// -- 득점시에만 candidate를 반환목록에 추가합니다.
			// 득점에 실패한 경우 candidate 메모리를 재활용합니다.(else)
			if (bGotScore)
			{
				res.Candidates.Add(cand);
				cand = null; // 소유권 이전합니다.
			}
			else
			{
				cand.Votes = 0.0f;
				foreach (var candBall in cand.Balls)
					candBall.Nodes.Clear();
			}
		}

		// 끝
		return res;
	}

	private void initSim()
	{
		_sim = new PhysContext();

		var (rst, w, h, f) = (_p.Table.Restitution, _p.Table.Width * 0.5f, _p.Table.Height * 0.5f, _p.Table.Friction);
		_wallPositions = new[]
		{
			(w, 0f), (-w, 0f), (w * 1.05f, 0f), (-w * 1.05f, 0f),
			(0f, h), (0f, -h), (0f, h * 1.05f), (0f, h * -1.05f)
		};

		var spn = new PhysStaticPlane();
		spn.RestitutionCoeff = rst;
		spn.DampingCoeff = f;

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
			_ballIndexes[index] = ballIndex;
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
