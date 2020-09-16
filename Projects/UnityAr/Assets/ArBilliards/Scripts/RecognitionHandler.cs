using System.Collections;
using System.Collections.Generic;
using System;
using System.CodeDom.Compiler;
using System.Globalization;
using System.Linq;
using System.Threading;
using UnityEngine;

public class RecognitionHandler : MonoBehaviour
{
	#region Exposed Properties

	public Transform tableTransform;
	public Transform red1;
	public Transform red2;
	public Transform orange;
	public Transform white;

	// 두 공 사이의 거리가 가깝다고 판단하는 거리입니다.
	public float nearbyThreshold = 0.01f;
	public float maxSpeed = 2.0f;
	public float errorCorrectionRate = 0.1f;
	public float speedDampingOnInvisibleState = 2.0f;
	#endregion

	#region Internal Fields

	private DateTime?[] _latestUpdates = new DateTime?[4];
	private Vector3[] _velocities = new Vector3[4];
	private Vector3?[] _prevPositions = new Vector3?[4];

	#endregion

	[Serializable]
	public struct RecognitionResult
	{
		[Serializable]
		public struct TableRecognitionDesc
		{
			public float[] Translation;
			public float[] Orientation;
			public float Confidence;
		}

		public TableRecognitionDesc Table;

		[Serializable]
		public struct BallRecognitionDesc
		{
			public float[] Position;
			public float Confidence;
		}

		public BallRecognitionDesc Red1;
		public BallRecognitionDesc Red2;
		public BallRecognitionDesc Orange;
		public BallRecognitionDesc White;
	}

	// Start is called before the first frame update
	private void Start()
	{

	}

	// Update is called once per frame
	private void Update()
	{
		// 최근에 제대로 갱신된 오브젝트에 대해서만 위치 추정을 수행합니다.
		var ballTrs = new[] { red1, red2, orange, white };
		for (int index = 0; index < 4; index++)
		{
			if (!_latestUpdates[index].HasValue)
			{
				_velocities[index] -= _velocities[index] * speedDampingOnInvisibleState * Time.deltaTime;
			}
			{
				var ballTr = ballTrs[index];
				ballTr.position += _velocities[index] * Time.deltaTime;
			}
		}
	}

	static Vector3 toVector3(RecognitionResult.BallRecognitionDesc Desc)
	{
		return new Vector3(Desc.Position[0], Desc.Position[1], Desc.Position[2]);
	}

	public void UpdateRecognition(RecognitionResult result)
	{
		if (tableTransform)
		{
			if (result.Table.Confidence > 0.5f)
			{
				var vec = new Vector3();
				vec.x = result.Table.Translation[0];
				vec.y = result.Table.Translation[1];
				vec.z = result.Table.Translation[2];
				tableTransform.localPosition = vec;

				// Orientation은 Rodrigues 표현식을 사용하므로, 축 및 회전으로 표현합니다.
				vec.x = result.Table.Orientation[0];
				vec.y = result.Table.Orientation[1];
				vec.z = result.Table.Orientation[2];

				var rot = Quaternion.AngleAxis(vec.magnitude * 180.0f / (float)Math.PI, vec.normalized);

				// 만들어진 rotation으로부터, yaw 회전을 제외한 나머지 회전을 suppress합니다.
				// var euler = rot.eulerAngles;
				// euler.z = euler.x = 0;
				// rot.eulerAngles = euler;
				tableTransform.localRotation = rot;
			}
		}

		{
			var balls = new[] { red1, red2, orange, white };
			var actualIndex = new[] { 0, 1, 2, 3 };
			var ballResults = new[] { result.Red1, result.Red2, result.Orange, result.White };
			bool bRedSwap = false;

			// 더 가까운 것을 가져갑니다. 
			// 가중치는 속도 벡터의 차이와 위치 벡터의 차이로 계산됩니다.
			if (result.Red1.Confidence > 0.5f)
			{
				if (result.Red2.Confidence > 0.5f)
				{
					// 가장 오차가 작은 쌍을 찾습니다.
					var aDist1 = (toVector3(result.Red1) - red1.position).magnitude;
					var aDist2 = (toVector3(result.Red2) - red1.position).magnitude;
					var bDist1 = (toVector3(result.Red1) - red2.position).magnitude;
					var bDist2 = (toVector3(result.Red2) - red2.position).magnitude;

					var array = new[] { aDist1, aDist2, bDist1, bDist2 };
					var minValue = array.Min();

					if (minValue == aDist2 || minValue == bDist1)
					{
						bRedSwap = true;
					}
				}
				else
				{
					var dist1 = (toVector3(result.Red1) - red1.position).magnitude;
					var dist2 = (toVector3(result.Red1) - red2.position).magnitude;

					if (dist2 < dist1)
					{
						bRedSwap = true;
					}
				}
			}

			if (bRedSwap)
			{
				actualIndex[0] = 1;
				actualIndex[1] = 0;
			}

			for (int index = 0; index < 4; index++)
			{
				var aidx = actualIndex[index];
				var ballTr = balls[index];
				var ballResult = ballResults[aidx];
				Vector3 ballPos = toVector3(ballResult);

				if (ballResult.Confidence > 0.5f)
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
						var velocity = (deltaPosition + errorCorrection) / deltaTime;

						if (velocity.magnitude > maxSpeed)
						{
							velocity = velocity.normalized * maxSpeed;
						}
						_velocities[index] = velocity; // velocity.magnitude > maxSpeed ? Vector3.zero : velocity;
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
					// _prevPositions[velIndex] = null;
				}
			}
		}
	}
}



