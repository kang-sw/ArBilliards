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
using UnityEngine.Rendering;

public class JsonIPC : MonoBehaviour
{
	public UInt16 CmdPort;
	public UInt16 BinPort;
	public string IpAddr = "localhost";
	public Transform TrackingTransform;
	public float UpdatePeriod = 1.0f;
	public ZEDManager Zed;

	public Transform TableTransformRaw;

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

		public async void Connect(JsonIPC Host, string IpAddr, ushort Port)
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

			try
			{
				await ConnectTask;
				ConnectTask = null;
				WrB = new BinaryWriter(Socket.GetStream());
				RdB = new BinaryReader(Socket.GetStream());
				WrS = new StreamWriter(Socket.GetStream());
				RdS = new StreamReader(Socket.GetStream());

				Host.OnConnect();
			}
			catch
			{

			}
		}

		public void Close()
		{
			Socket?.Close();
		}
	}

	Conn Cmd;
	Conn Bin;

	public Texture2D Pixels;
	public Texture2D Depths;
	int StampGen = 0;

	bool bProcessingAsyncReadback;
	byte[] ProcessingPixelBuf;
	byte[] ProcessingDepthBuf;

	string JsonRecognitionResult;

	[Serializable]
	struct JsonPacket
	{
		public int Stamp;

		public float[] Translation;
		public float[] Orientation;

		public int RgbW;
		public int RgbH;

		public int DepthW;
		public int DepthH;

		public JsonCameraParams Camera;
	}

	[Serializable]
	struct JsonCameraParams
	{
		public double fx, fy, cx, cy;
		public double k1, k2, p1, p2;
	}

	JsonCameraParams? CameraParamCache;

	[Serializable]
	struct JsonRecognitionReturnArg
	{
		[Serializable]
		public struct TableRecognitionArg
		{
			public float[] Translation;
			public float[] Orientation;
			public float Confidence;
		}

		public TableRecognitionArg Table;
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
		if (JsonRecognitionResult != null)
		{
			var JsonStr = JsonRecognitionResult;
			JsonRecognitionResult = null;
			Debug.Log(JsonStr);

			var Result = JsonUtility.FromJson<JsonRecognitionReturnArg>(JsonStr);

			if (TableTransformRaw)
			{
				if (Result.Table.Confidence > 0.5f)
				{
					var pos = new Vector3();
					pos.x = Result.Table.Translation[0];
					pos.y = Result.Table.Translation[1];
					pos.z = Result.Table.Translation[2];
					TableTransformRaw.localPosition = pos;
				}
			}
		}

		if ((PeriodTimer += Time.deltaTime) < UpdatePeriod)
		{
			return;
		}
		PeriodTimer -= UpdatePeriod;

		if (!Cmd.Connected)
		{
			Cmd.Connect(this, IpAddr, CmdPort);
		}

		if (!Bin.Connected)
		{
			Bin.Connect(this, IpAddr, BinPort);
		}

		if (!Cmd.Connected || !Bin.Connected)
		{
			CameraParamCache = null;
			return;
		}

		if (TrackingTransform && Zed.IsZEDReady && !bProcessingAsyncReadback)
		{
			// Debug.Log(JsonUtility.ToJson(Tracking));
			var O = new JsonPacket();
			O.Stamp = StampGen++;

			var pos = TrackingTransform.position;
			var rot = TrackingTransform.rotation.eulerAngles;
			O.Translation = new float[] { pos.x, -pos.y, pos.z };
			O.Orientation = new float[] { rot.x, -rot.y, rot.z };

			if (Pixels == null)
				Pixels = Zed.zedCamera.CreateTextureImageType(VIEW.LEFT);
			if (Depths == null)
				Depths = Zed.zedCamera.CreateTextureMeasureType(MEASURE.DEPTH);

			O.RgbW = Pixels.width;
			O.RgbH = Pixels.height;
			O.DepthW = Depths.width;
			O.DepthH = Depths.height;

			if (CameraParamCache == null)
			{
				var c = new JsonCameraParams();
				var cm = Zed.zedCamera.CalibrationParametersRectified.leftCam;
				c.cx = cm.cx;
				c.cy = cm.cy;
				c.fx = cm.fx;
				c.fy = cm.fy;
				if (cm.disto != null)
				{
					c.k1 = cm.disto[0];
					c.k2 = cm.disto[1];
					c.p1 = cm.disto[2];
					c.p2 = cm.disto[3];
				}

				CameraParamCache = c;
			}

			O.Camera = CameraParamCache.Value;

			Cmd.WrS.Write(JsonUtility.ToJson(O));
			Cmd.WrS.Write((char)3);
			Cmd.WrS.Flush();

			int Stamp = O.Stamp;

			bProcessingAsyncReadback = true;

			var PixelCallback = new Action<AsyncGPUReadbackRequest>((AsyncGPUReadbackRequest d) =>
			{
				ProcessingPixelBuf = d.GetData<byte>().ToArray();

				TryCheckSendBinaryBuf(Stamp);
			});

			var DepthCallback = new Action<AsyncGPUReadbackRequest>((AsyncGPUReadbackRequest d) =>
			{
				ProcessingDepthBuf = d.GetData<byte>().ToArray();

				TryCheckSendBinaryBuf(Stamp);
			});

			var PixelReadRequest = AsyncGPUReadback.Request(Pixels, 0, PixelCallback);
			var DepthReadRequest = AsyncGPUReadback.Request(Depths, 0, DepthCallback);

			try
			{
				Cmd.RdS.ReadLineAsync().ContinueWith(OnAsyncResultJsonLineRecv);
			}
			catch (InvalidOperationException) { }
		}
	}

	void OnConnect()
	{
		bProcessingAsyncReadback = false;
		ProcessingPixelBuf = ProcessingDepthBuf = null;

	}

	void OnAsyncResultJsonLineRecv(Task<string> Op)
	{
		var JsonStr = Op.Result;
		JsonRecognitionResult = JsonStr;
	}

	void TryCheckSendBinaryBuf(int Stamp)
	{
		if (ProcessingDepthBuf != null && ProcessingPixelBuf != null)
		{
			new Task(() =>
			{
				Bin.WrB.Write(0x00abcdef);
				Bin.WrB.Write(Stamp);

				Bin.WrB.Write(ProcessingPixelBuf.Length);
				Bin.WrB.Write(ProcessingDepthBuf.Length);
				Bin.WrB.Write(ProcessingPixelBuf.ToArray());
				Bin.WrB.Write(ProcessingDepthBuf.ToArray());
				Bin.WrB.Flush();

				bProcessingAsyncReadback = false;
				ProcessingDepthBuf = ProcessingPixelBuf = null;
			}).Start();
		}
	}
}

