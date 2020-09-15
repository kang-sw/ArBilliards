using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

public class RecognitionHandler : MonoBehaviour
{
	#region Exposed Properties

	public Transform tableTransform;

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
	}
}
