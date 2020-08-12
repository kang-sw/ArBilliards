using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using UnityEngine;
using System.Linq;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using sl;
using Oculus.Platform;

public class JsonIPC : MonoBehaviour
{
	public UInt16 CmdPort;
	public UInt16 BinPort;
	public string IpAddr = "localhost";
	public Transform TrackingTransform;
	public float UpdatePeriod = 1.0f;
	public ZEDManager Zed;

	float PeriodTimer = 0;

	public class Conn
	{
		public TcpClient Socket { get; private set; }
		public BinaryWriter WrB { get; private set; }
		public BinaryReader RdB { get; private set; }
		public StreamWriter WrS { get; private set; }
		public StreamReader RdS { get; private set; }

		Task ConnectTask;
		ushort Port;

		public bool Connected
		{
			get { return Socket != null && Socket.Connected; }
		}

		public async void Connect(string IpAddr, ushort Port)
		{
			this.Port = Port;

			if (ConnectTask?.IsFaulted == true)
			{
				ConnectTask = null;
			}

			if (Connected)
			{	
				return;
			}

			if (ConnectTask != null)
			{
				return;
			}

			Socket = new TcpClient();
			ConnectTask = Socket.ConnectAsync(IpAddr, Port);

			await ConnectTask;
			ConnectTask = null;
			WrB = new BinaryWriter(Socket.GetStream());
			RdB = new BinaryReader(Socket.GetStream());
			WrS = new StreamWriter(Socket.GetStream());
			RdS = new StreamReader(Socket.GetStream());
		}

		public void Close()
		{
			Socket?.Close();
		}
	}

	public Conn Cmd { get; private set; }
	public Conn Bin { get; private set; }

	Texture2D Pixels;
	Texture2D Depths;
	int StampGen = 0;

	[Serializable]
	struct TestJsonObject
	{
		public int Stamp;

		public float[] Translation;
		public float[] Orientation;

		public int ImageW;
		public int ImageH;
	}

	// Start is called before the first frame update
	void Start()
	{
		Cmd = new Conn();
		Bin = new Conn();
	}

	private void OnDestroy()
	{
		Cmd.Close();
		Bin.Close();
	}

	// Update is called once per frame
	void Update()
	{
		if ((PeriodTimer += Time.deltaTime) < UpdatePeriod)
		{
			return;
		}
		PeriodTimer -= UpdatePeriod;

		if (!Cmd.Connected)
		{
			Cmd.Connect(IpAddr, CmdPort);
		}

		if (!Bin.Connected)
		{
			Bin.Connect(IpAddr, BinPort);
		}

		if (!Cmd.Connected || !Bin.Connected)
		{
			return;
		}

		if (TrackingTransform && Zed.IsZEDReady)
		{
			// Debug.Log(JsonUtility.ToJson(Tracking));
			var JsonObj = new TestJsonObject();
			JsonObj.Stamp = StampGen++;

			var pos = TrackingTransform.position;
			var rot = TrackingTransform.rotation.eulerAngles;
			JsonObj.Translation = new float[] { pos.x, -pos.y, pos.z };
			JsonObj.Orientation = new float[] { rot.x, -rot.y, rot.z };

			if (Pixels == null)
			{
				Pixels = Zed.zedCamera.CreateTextureImageType(VIEW.LEFT);
			}
			if (Depths == null)
			{
				Depths = Zed.zedCamera.CreateTextureMeasureType(MEASURE.DEPTH);
			}

			JsonObj.ImageW = Pixels.width;
			JsonObj.ImageH = Pixels.height;

			Cmd.WrS.Write(JsonUtility.ToJson(JsonObj));
			Cmd.WrB.Write((char)0);
			Cmd.WrS.Flush();

			Bin.WrB.Write(0x00abcdef);
			Bin.WrB.Write(JsonObj.Stamp);

			var PixelBuf = Pixels.GetRawTextureData();
			var DepthBuf = Pixels.GetRawTextureData();
			Bin.WrB.Write(PixelBuf.Length);
			Bin.WrB.Write(DepthBuf.Length);
			Bin.WrB.Write(PixelBuf);
			Bin.WrB.Write(DepthBuf);
			Bin.WrB.Flush();
		}
	}
}

