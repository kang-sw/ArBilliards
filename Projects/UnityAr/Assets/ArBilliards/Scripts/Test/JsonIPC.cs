using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using UnityEngine;
using System.Linq;
using System.IO;

public class JsonIPC : MonoBehaviour
{
	public UInt16 PortNumber = 25033;
	public string IpAddr = "localhost";
	public Transform TrackingTransform;
	public float UpdatePeriod = 1.0f;

	float PeriodTimer = 0;
	TcpClient Connection;
	StreamWriter NetWrite;
	System.Threading.Tasks.Task AsyncConnectTask;

	[Serializable]
	class TestJsonObject
	{
		public float[] Translation;
		public float[] Orientation;
	}


	// Start is called before the first frame update
	void Start()
	{
		Connection = new TcpClient();
	}

	private void OnDestroy()
	{
		if(Connection.Connected)
		{
			Connection.Close();
		}
	}

	// Update is called once per frame
	void Update()
	{
		if ((PeriodTimer += Time.deltaTime) < UpdatePeriod)
		{
			return;
		}
		PeriodTimer -= UpdatePeriod;

		if (!Connection.Connected)
		{
			if (AsyncConnectTask == null || AsyncConnectTask.Status == System.Threading.Tasks.TaskStatus.Faulted)
			{
				NetWrite = null;
				Connection = new TcpClient();
				AsyncConnectTask = Connection.ConnectAsync(IpAddr, PortNumber);
			}

			return;
		}

		AsyncConnectTask = null;

		if (NetWrite == null)
		{
			NetWrite = new StreamWriter(Connection.GetStream());
		}

		if (TrackingTransform)
		{
			// Debug.Log(JsonUtility.ToJson(Tracking));
			var JsonObj = new TestJsonObject();
			var pos = TrackingTransform.position;
			var rot = TrackingTransform.rotation.eulerAngles;
			JsonObj.Translation = new float[] { pos.x, -pos.y, pos.z };
			JsonObj.Orientation = new float[] { rot.x, -rot.y, rot.z };
			NetWrite.WriteLine(JsonUtility.ToJson(JsonObj));
			NetWrite.Write((char)3);
			NetWrite.Flush();
		}
	}
}

