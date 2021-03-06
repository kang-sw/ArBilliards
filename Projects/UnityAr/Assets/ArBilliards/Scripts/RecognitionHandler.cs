﻿using System;
using System.Linq;
using UnityEngine;

public class RecognitionHandler : MonoBehaviour
{
	#region Exposed Properties

	[Header("Transforms")]
	public Transform TableTr;
	public Transform Red1;
	public Transform Red2;
	public Transform Orange;
	public Transform White;

	public Transform Red1FeltContact;
	public Transform Red2FeltContact;
	public Transform OrnageFeltContact;
	public Transform WhiteFeltContact;

	public Transform[] TablePillarTransforms;

	public Transform CameraAnchorOffset;

	// 테이블을 기준으로 정정될 트랜스폼 목록입니다.
	public Transform AdjustedTransform;

	[Header("Recognition Parameters")]
	// 두 공 사이의 거리가 가깝다고 판단하는 거리입니다.
	public float nearbyThreshold = 0.01f;
	public float maxSpeed = 2.0f;
	public float errorCorrectionRate = 0.1f;
	public float speedDampingOnInvisibleState = 2.0f;
	public float stopStanceFilterCoeff = 0.27f;

	// 공이 멈췄다고 판단하는 속도입니다.
	public float stopSpeed = 0.01f;

	[Header("Simulator")]
	// 시뮬레이터 레퍼런스
	public SimHandler Simulator;

	[Header("Environment Parameter Apply Target")]
	public Renderer leftEyeframeMatRenderer;
	public Renderer rightEyeframeMatRenderer;

	#endregion

	#region Internal Fields

	private DateTime?[] _latestUpdates = new DateTime?[4];
	private Vector3[] _velocities = new Vector3[4];
	private Vector3?[] _prevPositions = new Vector3?[4];
	private Vector3[] _positionFilteredOnStop = new Vector3[4];
	private static readonly int TableHSVH = Shader.PropertyToID("_TableHSV_H");
	private static readonly int TableHSVS = Shader.PropertyToID("_TableHSV_S");

	#endregion

	[Serializable]
	public struct RecognitionResult
	{
		[Serializable]
		public class TableRecognitionDesc
		{
			public float[] Translation;
			public float[] Orientation;
			public float Confidence;
		}

		[Serializable]
		public class TablePropertyDesc
		{
			public float InnerWidth;
			public float InnerHeight;

			public float ShaderMinH;
			public float ShaderMaxH;
			public float ShaderMinS;
			public float ShaderMaxS;

			public bool EnableShaderApplyDepthOverride;
		}

		public TableRecognitionDesc Table;
		public TablePropertyDesc TableProps;

		[Serializable]
		public class BallRecognitionDesc
		{ 
			public float[] Position;
			public float Confidence;
		}
		public BallRecognitionDesc Red1;
		public BallRecognitionDesc Red2;
		public BallRecognitionDesc Orange;
		public BallRecognitionDesc White;

		public float BallRadius;

		[Serializable]
		public class PhysDesc
		{
			public float BallRestitution;
			public float BallDamping;
			public float BallStaticFriction;
			public float BallRollTime;
			public float TableRestitution;
			public float TableRtoVCoeff;
			public float TableVtoRCoeff;
		}

		public PhysDesc Phys;
		public float[] CameraAnchorOffset;
	}

	// Start is called before the first frame update
	private void Start()
	{
	}

	// Update is called once per frame
	private void Update()
	{
		// 최근에 제대로 갱신된 오브젝트에 대해서만 위치 추정을 수행합니다.
		var ballTrs = new[] { Red1, Red2, Orange, White };
		for (int index = 0; index < 4; index++)
		{
			if (!_latestUpdates[index].HasValue)
			{
				_velocities[index] -= _velocities[index] * (speedDampingOnInvisibleState * Time.deltaTime);
			}
			{
				var ballTr = ballTrs[index];
				ballTr.position += _velocities[index] * Time.deltaTime;

				{
					_positionFilteredOnStop[index] = ballTr.position;
				}
			}

			Simulator.ReportedBallPositions = BallTransformWorldToLocal(_positionFilteredOnStop);
		}

		{
			var worldPositions = new Vector3[] { Red1.position, Red2.position, Orange.position, White.position };
			var pos = BallTransformWorldToLocal(worldPositions);
			var localPositions = new Vector3[] { pos.red1, pos.red2, pos.orange, pos.white };
			var trs = new Transform[] { Red1FeltContact, Red2FeltContact, OrnageFeltContact, WhiteFeltContact };

			for (int i = 0; i < 4; i++)
			{
				if (trs[i])
				{
					trs[i].localPosition = localPositions[i] + new Vector3(0, -Simulator.BallRadius, 0);
				}
			}
		}
	}

	static Vector3 toVector3(RecognitionResult.BallRecognitionDesc Desc)
	{
		return new Vector3(Desc.Position[0], Desc.Position[1], Desc.Position[2]);
	}

	public void UpdateRecognition(RecognitionResult recog)
	{
		UpdateTableTransform(ref recog);
		UpdateBallTransforms(ref recog);

		// 시뮬레이션을 수행 가능 여부를 질의합니다.
		// 모든 엘리먼트가 멈춰 있고, 위치가 제대로 인식된 경우입니다.
		bool bSimulationAvailable = true;
		while (recog.Table != null && recog.Red1 != null)
		{
			var confidenceArray = new[]
			{
				recog.Table.Confidence, (recog.Red1.Confidence +  recog.Red2.Confidence) * 0.7f, recog.Orange.Confidence,
				recog.White.Confidence
			};

			// 모든 필드 요소가 제대로 인식되었을 때에만 시뮬레이션 수행 가능
			if (confidenceArray.Min() < 0.5f)
			{
				bSimulationAvailable = false;
				break;
			}

			foreach (var velocity in _velocities)
			{
				if (velocity.magnitude > stopSpeed)
				{
					bSimulationAvailable = false;
					break;
				}
			}

			break;
		}

		// if (true && bSimulationAvailable && Simulator)
		{
			// var (red1, red2, orange, white) = (toVector3(result.Red1), toVector3(result.Red2), toVector3(result.Orange), toVector3(result.White));
			// var (red1, red2, orange, white) = (this.Red1.position, this.Red2.position, this.Orange.position, this.White.position);
			var res = BallTransformWorldToLocal(_positionFilteredOnStop);

			Simulator.SimulationBallPositions = res;
		}

		// 설정 등을 복사해옵니다.

		if (recog.CameraAnchorOffset != null)
		{
			if (TablePillarTransforms != null)
			{
				var w = recog.TableProps.InnerWidth * .5f;
				var h = recog.TableProps.InnerHeight * .5f;
				var offsets = new Vector3[]
				{
					new Vector3(w, 0, h),
					new Vector3(-w, 0, h),
					new Vector3(w, 0, -h),
					new Vector3(-w, 0, -h),
				};

				for (int i = 0; i < Math.Min(TablePillarTransforms.Length, offsets.Length); i++)
				{
					TablePillarTransforms[i].localPosition = offsets[i];
				}
			}

			Simulator.BallRadius = recog.BallRadius;
			Simulator.TableWidth = recog.TableProps.InnerWidth;
			Simulator.TableHeight = recog.TableProps.InnerHeight;

			if (leftEyeframeMatRenderer != null)
			{
				leftEyeframeMatRenderer.sharedMaterial.SetInt(
					"_EnableTableDepthOverride", recog.TableProps.EnableShaderApplyDepthOverride ? 1 : 0);
				leftEyeframeMatRenderer.sharedMaterial.SetVector(
					TableHSVH, new Vector4(recog.TableProps.ShaderMinH, recog.TableProps.ShaderMaxH));
				leftEyeframeMatRenderer.sharedMaterial.SetVector(
					TableHSVS, new Vector4(recog.TableProps.ShaderMinS, recog.TableProps.ShaderMaxS));
			}

			if (rightEyeframeMatRenderer != null)
			{
				rightEyeframeMatRenderer.sharedMaterial.SetInt(
					"_EnableTableDepthOverride", recog.TableProps.EnableShaderApplyDepthOverride ? 1 : 0);
				rightEyeframeMatRenderer.sharedMaterial.SetVector(
					TableHSVH, new Vector4(recog.TableProps.ShaderMinH, recog.TableProps.ShaderMaxH));
				rightEyeframeMatRenderer.sharedMaterial.SetVector(
					TableHSVS, new Vector4(recog.TableProps.ShaderMinS, recog.TableProps.ShaderMaxS));
			}

			{
				var phys = recog.Phys;
				Simulator.BallRestitution = phys.BallRestitution;
				Simulator.BallDamping = phys.BallDamping;
				Simulator.BallStaticFriction = phys.BallStaticFriction;
				Simulator.BallRollTime = phys.BallRollTime;
				Simulator.TableRestitution = phys.TableRestitution;
				Simulator.TableRollFriction = phys.TableRtoVCoeff;
				Simulator.TableVelocityFriction = phys.TableVtoRCoeff;
			}
		}


		if (recog.CameraAnchorOffset != null && recog.CameraAnchorOffset.Length == 3 && CameraAnchorOffset)
		{
			var a = recog.CameraAnchorOffset;
			CameraAnchorOffset.localPosition = new Vector3(a[0], a[1], a[2]);
		}
	}

	private (Vector3 red1, Vector3 red2, Vector3 orange, Vector3 white) BallTransformWorldToLocal(Vector3[] p)
	{
		var (red1, red2, orange, white) = (p[0], p[1], p[2], p[3]);
		var trs = TableTr.worldToLocalMatrix;

		void doTr(ref Vector3 vec)
		{
			var v = new Vector4(vec.x, vec.y, vec.z, 1.0f);
			v = trs * v;
			v.y = 0f;
			vec = v;
		}

		// 테이블 벡터에 대해 2D 좌표로 만들고, 시뮬레이션을 트리거합니다.
		doTr(ref red1);
		doTr(ref red2);
		doTr(ref orange);
		doTr(ref white);

		return (red1, red2, orange, white);
	}

	private void UpdateBallTransforms(ref RecognitionResult result)
	{
		if (result.Red1 == null || result.Red1.Position == null)
		{
			return;
		}

		var balls = new[] { Red1, Red2, Orange, White };
		var ballContacts = new[] { Red1FeltContact, Red2FeltContact, OrnageFeltContact, WhiteFeltContact };
		var actualIndex = new[] { 0, 1, 2, 3 };
		bool bRedSwap = false;

		// 더 가까운 것을 가져갑니다. 
		// 가중치는 속도 벡터의 차이와 위치 벡터의 차이로 계산됩니다.
		if (result.Red1.Confidence > 0f)
		{
			if (result.Red2.Confidence > 0f)
			{
				// 가장 오차가 작은 쌍을 찾습니다.
				var aDist1 = (toVector3(result.Red1) - Red1.position).magnitude;
				var aDist2 = (toVector3(result.Red2) - Red1.position).magnitude;
				var bDist1 = (toVector3(result.Red1) - Red2.position).magnitude;
				var bDist2 = (toVector3(result.Red2) - Red2.position).magnitude;

				var array = new[] { aDist1, aDist2, bDist1, bDist2 };
				var minValue = array.Min();

				if (minValue == aDist2 || minValue == bDist1)
				{
					bRedSwap = true;
				}
			}
			else
			{
				var dist1 = (toVector3(result.Red1) - Red1.position).magnitude;
				var dist2 = (toVector3(result.Red1) - Red2.position).magnitude;

				if (dist2 < dist1)
				{
					bRedSwap = true;
				}
			}
		}

		if (bRedSwap)
		{
			var tmp = result.Red1;
			result.Red1 = result.Red2;
			result.Red2 = tmp;
		}

		var ballResults = new[] { result.Red1, result.Red2, result.Orange, result.White };
		for (int index = 0; index < 4; index++)
		{
			var aidx = actualIndex[index];
			var ballTr = balls[index];
			var ballResult = ballResults[index];
			Vector3 ballPos = toVector3(ballResult);

			if (ballResult.Confidence > 0f)
			{
				// ballTr.position = ballPos;
				var now = DateTime.Now;
				if (_latestUpdates[index].HasValue)
				{
					var prevPosition = _prevPositions[index].Value;
					var deltaPosition = ballPos - prevPosition;
					var errorCorrection = (ballPos - ballTr.position) * errorCorrectionRate;

					var deltaTimeSpan = now - _latestUpdates[index];
					var deltaTime = deltaTimeSpan.Value.Milliseconds / 1000.0f;
					var velocity = ((deltaPosition * (1 - errorCorrectionRate)) + errorCorrection) / deltaTime;

					if (velocity.magnitude > maxSpeed)
					{
						velocity = velocity.normalized * maxSpeed;
					}

					_velocities[index] = velocity;
				}
				else
				{
					ballTr.position = ballPos;
				}

				_latestUpdates[index] = now;
				_prevPositions[index] = ballPos;
			}
			else
			{
				_latestUpdates[index] = null;
				_prevPositions[index] = null;
			}

		}

	}

	private void UpdateTableTransform(ref RecognitionResult result)
	{
		if (result.Table != null && result.Table.Confidence > 0f)
		{
			var vec = new Vector3();
			vec.x = result.Table.Translation[0];
			vec.y = result.Table.Translation[1];
			vec.z = result.Table.Translation[2];
			TableTr.localPosition = vec;

			// Orientation은 Rodrigues 표현식을 사용하므로, 축 및 회전으로 표현합니다.
			vec.x = result.Table.Orientation[0];
			vec.y = result.Table.Orientation[1];
			vec.z = result.Table.Orientation[2];

			var rot = Quaternion.AngleAxis(vec.magnitude * 180.0f / (float)Math.PI, vec.normalized);

			// 만들어진 rotation으로부터, yaw 회전을 제외한 나머지 회전을 suppress합니다.
			// var euler = rot.eulerAngles;
			// euler.z = euler.x = 0;
			// rot.eulerAngles = euler;
			TableTr.localRotation = rot;
		}
	}
}