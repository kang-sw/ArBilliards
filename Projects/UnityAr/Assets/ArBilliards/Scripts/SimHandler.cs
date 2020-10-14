using System;
using ArBilliards.Phys;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Debug = System.Diagnostics.Debug;

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
	public float SimInterval = 0.3f; // 시뮬레이션 간격입니다.
	public float PathRedrawInterval = 0.3f; // 경로 다시그리기 간격
	public float PathReplayInterval = 2.5f;
	public float PathForwardAngle = 30f; // 전방 방향으로 인식하는 각도

	[Header("Dimensions")]
	public float BallRadius = 0.07f / Mathf.PI;
	public float TableWidth = 0.96f;
	public float TableHeight = 0.51f;

	[Header("Coefficients")]
	public float BallRestitution = 0.71f;
	public float BallDamping = 0.33f;
	public float BallRollFriction = 0.01f;
	public float BallKineticFriction = 0.2f;
	public float BallStaticFriction = 0.12f;
	public float BallRollTime = 0.5f;
	public float TableRestitution = 0.53f;
	public float TableStaticFriction = 0.22f;
	public float TableHorizontalSuppress = 0.03f;

	[Header("Optimizations")]
	public int NumRotationDivider = 360;
	public float SimDuration = 10.0f;
	public float[] BallInitialSpeeds = new[] { 0.1f, 0.4f, 0.8f, 1.4f };
	public float ResimulationDistanceThreshold = 0.008f;
	public float MovementSpeedThreshold = 0.15f;

	[Header("GameState")]
	public BilliardsBall PlayerBall = BilliardsBall.Orange;

	[Header("Visualization")]
	public float CandidateMarkerLength = 0.2f;

	[Header("Visualizer Instances")]
	public Color[] BallVisualizeColors = new Color[4];
	public LineRenderer[] PathRenderers = new LineRenderer[4];

	[Header("Visualizer Templates")]
	public GameObject PathFollowMarkerTemplate;
	public GameObject CandidateRenderingTemplate;
	public GameObject CollisionMarkerTemplate;

	[Header("Rules")]
	public int NumMinCushions = 0;
	public bool Is3BallRule = false;

	#endregion

	#region Public States

	public (Vector3 Red1, Vector3 Red2, Vector3 Orange, Vector3 White)? SimulationBallPositions { get; set; }
	public (Vector3 Red1, Vector3 Red2, Vector3 Orange, Vector3 White) ReportedBallPositions { get; set; }

	#endregion

	#region Main Loop

	private GameObject _instancedRoot;
	private AsyncSimAgent.SimResult _latestResult = null;
	private bool _bLineDirty = false;
	private bool _prevMoveState;
	private bool _isAnyBallMoving = false;
	private bool _bShouldRetriggerSimulation = false;
	private (Vector3 Red1, Vector3 Red2, Vector3 Orange, Vector3 White) _latestSimPosition, _latestReportPosition;

	// Start is called before the first frame update
	void Start()
	{
		_instancedRoot = new GameObject("InstantiationRoot");
		_instancedRoot.transform.parent = TableAnchor;

		Start_AsyncSimulation();
		Start_Rendering();
	}

	// Update is called once per frame
	void Update()
	{
		float MaxDistance((Vector3 Red1, Vector3 Red2, Vector3 Orange, Vector3 White) prv,
			(Vector3 Red1, Vector3 Red2, Vector3 Orange, Vector3 White) opt)
		{
			var deltas = new[] { prv.Red1 - opt.Red1, prv.Red2 - opt.Red2, prv.Orange - opt.Orange, prv.White - opt.White };
			var dists = new float[4];

			for (var i = 0; i < deltas.Length; i++)
				dists[i] = deltas[i].magnitude;

			var maxDist = dists.Max();
			return maxDist;
		}

		// 만약 시뮬레이션 위치가 일정 이상 떨어졌다면 시뮬레이션 재실행합니다.
		// 이 때 너무 자주 재실행되지 않도록 유예 기간을 둡니다.
		{
			var rpt = ReportedBallPositions;
			var prvSim = _latestSimPosition;

			var simMaxDist = MaxDistance(rpt, prvSim);
			if (simMaxDist > ResimulationDistanceThreshold)
			{
				_bShouldRetriggerSimulation = true;
			}

			var prvRpt = _latestReportPosition;
			var rptMaxSpeed = MaxDistance(rpt, prvRpt) / Time.deltaTime;
			_latestReportPosition = rpt;
			_isAnyBallMoving = rptMaxSpeed > MovementSpeedThreshold;
		}

		Update_AsyncSimulation();
		Update_Rendering();

		_prevMoveState = _isAnyBallMoving;
	}

	void OnDestroy()
	{
		if (_parallelTask != null)
		{
			_parallelTask.Wait();
		}
	}

	void updateParam(ref AsyncSimAgent.InitParams p)
	{
		if (Is3BallRule)
		{
			AsyncSimAgent.InitParams.Rule3Balls(ref p);
		}
		else
		{
			AsyncSimAgent.InitParams.Rule4Balls(ref p);
		}

		p.SimDuration = SimDuration;
		p.Ball = (BallRestitution, BallDamping, BallRadius);
		p.PlayerBall = PlayerBall;
		p.Table = (TableRestitution, TableWidth, TableHeight, TableStaticFriction);
		p.BallFriction = (BallKineticFriction, BallRollFriction, BallStaticFriction);
		p.NumCandidates = NumRotationDivider;
		p.Speeds = (float[])BallInitialSpeeds.Clone();
		p.BallRollTime = BallRollTime;
		p.NumCushionHits = NumMinCushions;
		p.TableSuppress = TableHorizontalSuppress;
	}

	Vector3 getPlayerBallPosition()
	{
		return PlayerBall == BilliardsBall.Orange ? ReportedBallPositions.Orange : ReportedBallPositions.White;
	}

	#endregion

	#region Simulations

	private float _intervalTimer = 0f;
	private bool _bParallelProcessRunning;
	private Task _parallelTask;
	private AsyncSimAgent[] _parallel;
	private float AcceleratedDelta => _isAnyBallMoving ? 10.0f * Time.deltaTime : Time.deltaTime;

	void Start_AsyncSimulation()
	{
		// 다수의 스레드를 만듭니다.
		// 게임 스레드에 영향을 주지 않도록, 3개의 마진을 둡니다.
		// 적어도 하나는 생성되어야 합니다.
		_parallel = new AsyncSimAgent[Math.Max(1, SystemInfo.processorCount - 3)];

		for (var index = 0; index < _parallel.Length; index++)
		{
			_parallel[index] = new AsyncSimAgent();
		}
	}

	void Update_AsyncSimulation()
	{
		_intervalTimer += Time.deltaTime;

		if (_bShouldRetriggerSimulation)
		{
			_bShouldRetriggerSimulation = false;
			_intervalTimer = Math.Max(_intervalTimer, SimInterval - 0.5f);
		}

		if (_intervalTimer > SimInterval
			&& !_bParallelProcessRunning
			&& SimulationBallPositions.HasValue
			&& !_isAnyBallMoving)
		{
			_bParallelProcessRunning = true;

			// 파라미터 셋업
			var param = new AsyncSimAgent.InitParams();
			updateParam(ref param);


			(param.Red1, param.Red2, param.Orange, param.White) = _latestSimPosition = SimulationBallPositions.Value;
			SimulationBallPositions = null;

			// 비동기 시뮬레이션 트리거
			_parallelTask = new Task(() =>
			{
				var results = new AsyncSimAgent.SimResult();
				results.Options = param;

				Parallel.For(0, _parallel.Length, (index) =>
				{
					var agent = _parallel[index];
					var p = param;
					p.Parallel = (_parallel.Length, (int)index);

					var res = agent.InitSync(p);
					lock (results)
					{
						results.Candidates.AddRange(res.Candidates);
					}
				});

				_latestResult = results;
				_bLineDirty = true;
				_bParallelProcessRunning = false;
			});

			_parallelTask.Start();
			_intervalTimer = 0f;
		}
	}

	#endregion

	#region Visualizers

	private float _renderPeriodCounter = 0;
	private float _pathReplayCounter = 0;

	private List<MarkerManipulator> _collisionMarkerPool = new List<MarkerManipulator>();
	private int _numActiveCollisionMarkers;
	private int _cachedNumActiveCollisionMarkers;

	private MarkerManipulator[] _pathFollowMarker = new MarkerManipulator[4];

	private List<LineRenderer> _candidateMarkerPool = new List<LineRenderer>();
	private int _numActiveCandidateMarkers;

	private AsyncSimAgent.SimResult.Candidate _latestCandidate;

	private void Start_Rendering()
	{
		for (int i = 0; i < 4; i++)
		{
			var obj = Instantiate(PathFollowMarkerTemplate, TableAnchor);
			obj.transform.localScale = Vector3.one * BallRadius * 1.9f;
			_pathFollowMarker[i] = obj.GetComponent<MarkerManipulator>();
			var color = BallVisualizeColors[i];
			color.a = 0.66f;
			_pathFollowMarker[i].MeshColor = color;
			_pathFollowMarker[i].ParticleColor = color;

			// 패스팔로우 파티클 비활성화 (너무 요란함)
			var emission = _pathFollowMarker[i].ParticleSystem.emission;
			emission.enabled = false;
		}
	}

	private void Update_Rendering()
	{
		_renderPeriodCounter += AcceleratedDelta;
		_pathReplayCounter += Time.deltaTime;

		{
			// 공들이 움직이기 시작할 때와 멈출 때, 알파 값을 조정합니다.
			var newAlpha = _isAnyBallMoving ? 0.05f : 1f;
			foreach (var render in PathRenderers)
			{
				var col = render.startColor;
				col.a = Mathf.Lerp(col.a, newAlpha, Time.deltaTime * 2.0f);
				render.startColor = col;
			}
		}

		if (_latestResult != null
			&& _renderPeriodCounter > PathRedrawInterval
			&& LookTransform // 카메라가 있는지?
			&& _latestResult.Candidates.Count > 0
			&& !_isAnyBallMoving) // 계산된 결과가 존재하는지?
		{
			var fwd = LookTransform.forward;/*TableAnchor.localToWorldMatrix.MultiplyPoint(getPlayerBallPosition()) - LookTransform.position;*/
			var r = _latestResult;

			// 시뮬레이션이 갱신된 경우, 칠 라인을 갱신합니다.
			if (_bLineDirty)
			{
				_bLineDirty = false;
				_latestCandidate = null;
			}

			// -- 현재 카메라 방향에 따라 강조하기에 가장 적합한 candidate를 찾습니다.
			fwd = TableAnchor.worldToLocalMatrix.MultiplyVector(fwd);
			fwd.y = 0;
			AsyncSimAgent.SimResult.Candidate nearlest = r.Candidates[0];
			var distSorted = from elem in r.Candidates
							 where Vector3.Angle(elem.InitVelocity, fwd) < PathForwardAngle * 0.5f
							 orderby elem.InitVelocity.sqrMagnitude
							 select elem;

			nearlest = distSorted.FirstOrDefault();

			if (nearlest == null)
			{
				nearlest = r.Candidates[0];
				foreach (var elem in r.Candidates)
					if (Vector3.Angle(nearlest.InitVelocity, fwd) > Vector3.Angle(elem.InitVelocity, fwd))
						nearlest = elem;
			}

			if (_latestCandidate != null)
			{
				// Vote가 2배 이상 크지 않은 한, 기존의 선택을 유지.
				// if (nearlest.Votes < 2 * _latestCandidate.Votes)
				// {
				// 	nearlest = _latestCandidate;
				// }

				// 각도 차이가 
				if (Vector3.Angle(nearlest.InitVelocity, _latestCandidate.InitVelocity) < PathForwardAngle)
				{
					nearlest = _latestCandidate;
				}
			}

			// -- 모든 candidate에 대해 라인을 그립니다.
			// 오브젝트 풀 예약
			// 만약 nearlest element가 이전과 같다면 아래 과정을 모두 생략합니다.
			if (nearlest != _latestCandidate)
			{
				int numActive = 0;
				while (_candidateMarkerPool.Count < r.Candidates.Count + _numActiveCandidateMarkers)
				{
					var obj = Instantiate(CandidateRenderingTemplate, TableAnchor);
					var render = obj.GetComponentInChildren<LineRenderer>();
					render.positionCount = 2;
					obj.SetActive(false);
					_candidateMarkerPool.Add(render);
				}

				for (var i = 0; i < r.Candidates.Count; i++)
				{
					var elem = r.Candidates[i];
					var render = _candidateMarkerPool[i];
					var nodes = elem.Balls[(int)PlayerBall].Nodes;
					if (i >= _numActiveCandidateMarkers)
					{
						render.gameObject.SetActive(true);
						++_numActiveCandidateMarkers;
					}

					++numActive;
					var begin = nodes[0].Position;
					var end = nodes[1].Position;
					// var dir = nodes[1].Position - nodes[0].Position;
					// var end = dir.normalized * CandidateMarkerLength + begin;

					begin.y -= BallRadius * 1.01f;
					end.y -= BallRadius * 1.01f;
					render.SetPosition(0, begin);
					render.SetPosition(1, end);
				}


				// Trim inactive markers
				for (int i = numActive; i < _numActiveCandidateMarkers; ++i)
				{
					_candidateMarkerPool[i].gameObject.SetActive(false);
				}

				_numActiveCandidateMarkers = numActive;
				_pathReplayCounter += PathReplayInterval;
			}

			// -- 강조된 경로의 공을 그립니다.
			if (nearlest != _latestCandidate || _pathReplayCounter > PathReplayInterval)
			{
				_pathReplayCounter = 0f;
				initMarkerPool();
				for (int index = 0; index < 4; ++index)
					renderBallPath(PathRenderers[index], nearlest.Balls[index], true);
				trimUnusedMarkers();
			}

			// -- 가장 최근의 candidate 캐시 ...
			_latestCandidate = nearlest;
			_renderPeriodCounter = 0;
		}
	}

	void renderBallPath(LineRenderer target, AsyncSimAgent.BallPath path, bool shouldRestartPathFollow)
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
			var pos = nodes[index].Position;
			// pos.y -= BallRadius;

			target.SetPosition(index, pos);

			// 공과 접촉하는 각 조인트마다 충돌 마커를 스폰합니다.
			if ((index == 0 || nodes[index].Other.HasValue) && index < nodes.Count - 1)
			{
				var marker = spawnCollisionMarker();
				marker.ParticleColor = color;
				color.a = 0.765f;
				marker.MeshColor = color;
				marker.transform.localPosition = pos;
			}
		}

		// 패스 팔로우 활성화
		if (shouldRestartPathFollow)
		{
			var marker =
			_pathFollowMarker[(int)path.Index];
			marker.ActiveBallPath = path;
			marker.BallDamping = BallDamping;
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
			_collisionMarkerPool.Add(obj.GetComponent<MarkerManipulator>());
		}
	}

	MarkerManipulator spawnCollisionMarker()
	{
		var ret = _collisionMarkerPool[_cachedNumActiveCollisionMarkers];
		if (!ret.Active)
			ret.Active = true;

		++_cachedNumActiveCollisionMarkers;
		return ret;
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
		public (float Kinetic, float Roll, float Static) BallFriction;
		public float BallRollTime;
		public float TableSuppress;

		// RULES 
		public BilliardsBall PlayerBall; // 플레이어가 칠 공입니다.
		public bool bAvoidPlayerBallHit; // 4구 룰에서, 다른 플레이어의 공을 치면 실점 룰 적용
		public bool bOpponentBallAsScore; // 상대 공을 치는것을 점수로 칠건지 결정합니다. 3구 룰.
		public int NumCushionHits; // 마지막 공 타격 전까지의 최소 쿠션 히트 수입니다.

		// OPTIONS
		public float[] Speeds; // 최초 타구 시 속도입니다.
		public int SpeedDivisions; // 최초 타구시 속도를 몇 개의 uniform한 구간으로 나눌지 결정

		// OPTIMIZATION
		public int NumCandidates; // 360도 범위에서 몇 개의 후보를 선택할지 결정합니다. 후보는 uniform하게 선정됩니다.
		public (int Modulator, int Offset)? Parallel;
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
			public float Votes; // 해당 결과의 유용성 가중치
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
			public (BilliardsBall Index, Vector3 Position, Vector3 Velocity)? Other; // 충돌변인. null이면 벽에 충돌
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
			throw new Exception("Async process still running!");
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
			throw new Exception("Async process still running!");
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

		SimResult.Candidate cand = null;
		Stopwatch sw = new Stopwatch();
		sw.Start();

		foreach (var ballInitialSpeed in _p.Speeds) // 각각의 속도에 대해 Iteration ...
		{
			int candIndex, maxCands = _p.NumCandidates, step;

			if (_p.Parallel.HasValue)
				(candIndex, step) = (_p.Parallel.Value.Offset, _p.Parallel.Value.Modulator);
			else
				(candIndex, maxCands, step) = (0, _p.NumCandidates, 1);

			for (; candIndex < maxCands; candIndex += step)
			{
				if (sw.ElapsedMilliseconds > 2000)
				{
					break;
				}

				resetBallState();

				// -- 시뮬레이션 셋업
				cand = cand ?? new SimResult.Candidate(); // candidate를 찾는 데 실패한 경우 메모리를 재활용하기 위함입니다.
				var balls = cand.Balls;

				// -- 초기 속력 및 방향 지정
				float angle = (360f / maxCands) * candIndex;
				var dir = Quaternion.Euler(0, angle, 0) * Vector3.forward;

				var initVelocity = ballInitialSpeed * dir;
				cand.InitVelocity = initVelocity;
				_ballRefs[(int)_p.PlayerBall].SourceVelocity = initVelocity;

				// -- 공 초기 위치 노드 셋업
				for (int i = 0; i < 4; ++i)
				{
					var r = _ballRefs[i];

					BallPath.Node n;
					n.Position = r.SourcePosition;
					n.Velocity = r.SourceVelocity;
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
							if (B.HasValue)
								n.Other = (B.Value, ct.B.Pos, ct.B.Vel);
							else
								n.Other = null;

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
				float weight = 0; // 가중치 값으로, 더 조건이 좋은 공을 선별하는 데 사용합니다.

				foreach (var node in balls[(int)_p.PlayerBall].Nodes)
				{
					if (!node.Other.HasValue) // Other가 비었으면 벽입니다.
					{
						++numCushionHit;
					}
					else
					{
						var other = node.Other.Value;
						var index = (int)other.Index;

						// 4구 룰 한정
						int? otherBall = null;

						if (index == 0)
							otherBall = 1;
						else if (index == 1)
							otherBall = 0;

						// 다른 공을 치기 전 이미 친 공을 다시 치는 경우, vote를 감합니다.
						//if (otherBall.HasValue && hits[index].bHit != 0 && hits[otherBall.Value].bHit == 0)
						//{
						//	weight -= 1f;
						//} 

						// 만약 공을 때릴 때, 이미 

						if (hits[index].bHit == 0  // 처음 치는 공인 경우
							&& otherBall.HasValue) // otherBall이 있다는 것은 빨간 공을 때렸다는 뜻
						{
							// 충돌 각도가 클수록(접점-중점 방향 벡터와 속도의 내적으로 판단) 더 좋은 충돌입니다.
							var contactDir = (other.Position - node.Position).normalized;
							var angleWeight = Vector3.Dot(contactDir, node.Velocity.normalized);
							angleWeight = 2f * Mathf.Pow(angleWeight, 0.633f); // 내적이 0에 가까울수록 = 얇게 부딪칠수록 낮은 가중치
							weight += Math.Max(angleWeight, 0.98f);


							// 때린 공의 타임라인을 쫓아, 때리는 공이 다른 공과 몇번 충돌했는지 검사합니다.
							int otherIdx = otherBall.Value;
							var prevWallHits = from timestamp in balls[otherIdx].Nodes
											   where timestamp.Time < node.Time && timestamp.Other == null
											   select timestamp;
							var prevBallHits = from timestamp in balls[otherIdx].Nodes
											   where timestamp.Time < node.Time && timestamp.Other != null
											   select timestamp;
							weight -= prevWallHits.Count() * 1f + prevBallHits.Count() * 2f;
						}

						hits[index] = (1, numCushionHit);
					}
				}

				cand.Votes = weight;

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
		spn.Restitution = rst;
		spn.Damping = _p.TableSuppress;
		spn.Friction.Static = f;
		foreach (var pos in _wallPositions)
		{
			// 위치, 노멀 설정은 아래에서 ...
			spn.SourcePosition = new Vector3(pos.x, 0, pos.y);
			spn.Normal = new Vector3(-pos.x, 0, -pos.y);
			_sim.Spawn(spn);
		}

		// 공 스폰 및 초기 위치 목록 작성
		_p.Red1.y = _p.Red2.y = _p.Orange.y = _p.White.y = 0;
		_ballPositions = new[] { _p.Red1, _p.Red2, _p.Orange, _p.White };

		var ball = new PhysSphere();
		ball.Radius = _p.Ball.Radius;
		ball.Restitution = _p.Ball.Restitution;
		ball.Damping = _p.Ball.Damping;
		ball.Friction = _p.BallFriction;
		ball.RollBeginTime = _p.BallRollTime;

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
			ballRef.SourcePosition = _ballPositions[index];
			ballRef.SourceVelocity = Vector3.zero;
		}
	}
}
