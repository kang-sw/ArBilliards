using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionMarkerManipulator : MonoBehaviour
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

	public bool Active
	{
		get => gameObject.activeSelf;
		set => gameObject.SetActive(value);
	}
	 
	public MeshRenderer Mesh => GetComponentInChildren<MeshRenderer>();
	public ParticleSystem ParticleSystem => GetComponentInChildren<ParticleSystem>();


	// Start is called before the first frame update
	void Start()
	{
	}

	//// Update is called once per frame
	//void Update()
	//{

	//}

	// Update color change
	void OnValidate()
	{
		var color = effectColorEditor;


		Mesh.sharedMaterial.color = color;
		color.a = 1.0f;
		ParticleColor = color;
	}
}
