using System.Collections;
using System.Collections.Generic;
using System;
using System.CodeDom.Compiler;
using System.Globalization;
using UnityEngine;

public class RecognitionHandler : MonoBehaviour
{
	#region Exposed Properties

	public Transform tableTransform;
	public Transform red1;
	public Transform red2;
	public Transform orange;
	public Transform white;

	#endregion

	#region Internal Fields

	private DateTime?[] _latestUpdates = new DateTime?[4];
	private Vector3[] _velocities = new Vector3[4];

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
			if (_latestUpdates[index].HasValue)
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
			var velocityTargetIdx = new[] { 0, 1, 2, 3 };
			var ballResults = new[] { result.Red1, result.Red2, result.Orange, result.White };

			// red1인 특수한 경우로, 더 가까운 것을 가져갑니다.
			if (result.Red1.Confidence > 0.5)
			{
				float len1 = (toVector3(result.Red1) - red1.position).sqrMagnitude;
				float len2 = (toVector3(result.Red1) - red2.position).sqrMagnitude;

				if (len2 < len1)
				{
					balls[0] = red2;
					balls[1] = red1;
					velocityTargetIdx[0] = 1;
					velocityTargetIdx[1] = 0;
				}
			}

			for (int index = 0; index < 4; index++)
			{
				var ballTr = balls[index];
				var ballResult = ballResults[index];
				Vector3 ballPos = toVector3(ballResult);

				if (ballResult.Confidence > 0.5f)
				{
					var prev = ballTr.position;
					ballTr.position = Vector3.Lerp(ballTr.position, ballPos, 0.5f);
					var deltaPosition = ballTr.position - prev;

					var now = DateTime.Now;
					if (_latestUpdates[index].HasValue)
					{
						var deltaTimeSpan = now - _latestUpdates[index];
						var deltaTime = deltaTimeSpan.Value.Milliseconds / 1000.0f;
						var velocity = deltaPosition / deltaTime;
						_velocities[velocityTargetIdx[index]] = velocity;
					}

					_latestUpdates[index] = now;
				}
				else
				{
					_latestUpdates[index] = null;
				}
			}
		}
	}
}



