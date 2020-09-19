using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MarkerManipulator : MonoBehaviour
{

	[SerializeField] private Color effectColorEditor = Color.white;

	public Color MeshColor
	{
		get => Mesh.material.color;
		set => Mesh.material.color = value;
	}

	public Color ParticleColor
	{
		get => ParticleSystem.main.startColor.color;
		set {
			var main = ParticleSystem.main;
			main.startColor = value;
		}
	}

	public AsyncSimAgent.BallPath ActiveBallPath
	{
		get => _ballPath;
		set {
			_timeStamp = 0;
			_ballPath = value;
		}
	}

	public float BallDamping
	{
		get;
		set;
	}

	public bool Active
	{
		get => gameObject.activeSelf;
		set => gameObject.SetActive(value);
	}

	public MeshRenderer Mesh => GetComponentInChildren<MeshRenderer>();
	public ParticleSystem ParticleSystem => GetComponentInChildren<ParticleSystem>();

	private Transform _tr;
	private float _timeStamp;
	private AsyncSimAgent.BallPath _ballPath;

	// Start is called before the first frame update
	void Start()
	{
		_tr = transform;
	}

	//// Update is called once per frame
	void Update()
	{
		if (_ballPath != null)
		{
			var nodes = _ballPath.Nodes;
			_timeStamp += Time.deltaTime;

			for (var i = 0; i < _ballPath.Nodes.Count - 1; i++)
			{
				var beg = nodes[i];
				var end = nodes[i + 1];

				if (beg.Time < _timeStamp && _timeStamp < end.Time)
				{
					// var alpha = (_timeStamp - beg.Time) / (end.Time - beg.Time);
					// _tr.localPosition = Vector3.Lerp(beg.Position, end.Position, alpha);
					var t = _timeStamp - beg.Time;
					var alpha = BallDamping;
					var alpha_inv = 1.0 / BallDamping;
					_tr.localPosition = beg.Position + beg.Velocity * (float)(alpha_inv * (1 - Math.Exp(-alpha * t)));
					break;
				}
			}

			if (_timeStamp > nodes.Last().Time)
			{
				_timeStamp = 0f;
			}
		}

	}

	// Update color change
	void OnValidate()
	{
		var color = effectColorEditor;

		Mesh.sharedMaterial.color = color;
		color.a = 1.0f;
		ParticleColor = color;
	}
}
