using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PositionCorrector : MonoBehaviour
{
	public Transform ReadTransform;

	// Start is called before the first frame update
	void Start()
	{

	}

	// Update is called once per frame
	void Update()
	{
		if (ReadTransform)
		{
			transform.localPosition = -ReadTransform.position;
			transform.localRotation = Quaternion.Inverse(ReadTransform.localRotation);
		}
	}
}
