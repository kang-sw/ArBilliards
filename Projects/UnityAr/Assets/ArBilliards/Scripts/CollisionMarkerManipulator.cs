using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionMarkerManipulator : MonoBehaviour
{

	[SerializeField] private Color effectColorEditor = Color.white;

	public Color MeshColor
	{
		get => _mesh.material.color;
		set => _mesh.material.color = value;
	}

	public Color ParticleColor
	{
		get => _particle.main.startColor.color;
		set
		{
			var main = _particle.main;
			main.startColor = value;
		}
	}

	public bool Active
	{
		get => _bActive;
		set => gameObject.SetActive(value);
	}

	private bool _bActive;
	private MeshRenderer _mesh;
	private ParticleSystem _particle;


	// Start is called before the first frame update
	void Start()
	{
		_mesh = GetComponentInChildren<MeshRenderer>();
		_particle = GetComponentInChildren<ParticleSystem>();
	}

	//// Update is called once per frame
	//void Update()
	//{

	//}
	 
	// Update color change
	void OnValidate()
	{
		var color = effectColorEditor;
		_mesh = GetComponentInChildren<MeshRenderer>();
		_particle = GetComponentInChildren<ParticleSystem>();

		_mesh.sharedMaterial.color = color;
		color.a = 1.0f;
		ParticleColor = color;
	}
}
