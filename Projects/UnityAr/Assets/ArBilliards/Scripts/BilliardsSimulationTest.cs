using System.Collections;
using System.Collections.Generic;
using ArBilliards.Phys;
using UnityEngine;

public class BilliardsSimulationTest : MonoBehaviour
{
	public GameObject SphereTemplate;
	public GameObject PlaneTemplate;
	public Transform Anchor;
	public float SimulationStep = 0.5f;

	private PhysContext _sim = new PhysContext();
	private Dictionary<PhysObject, GameObject> _mapping = new Dictionary<PhysObject, GameObject>();
	private bool _manualSimMode = true;

	// Start is called before the first frame update
	void Start()
	{
		{
			var sph = new PhysSphere();
			sph.Mass = 10f;
			sph.DampingCoeff = 0.44;
			sph.RestitutionCoeff = 0.61f;
			  
			sph.Radius = 0.25f;
			sph.Velocity = new Vector3(1.4f, 0.0f);
			_sim.Spawn(sph);
			 
			sph.Position = new Vector3(0.8f, 0.0f);
			sph.Velocity = Vector3.zero;
			_sim.Spawn(sph);
			
			sph.Position = new Vector3(-0.3f, 0.1f);
			sph.Velocity = new Vector3(0.12f, -0.02f);
			_sim.Spawn(sph);
		}
		{
			var pln = new PhysStaticPlane();
			pln.RestitutionCoeff = 0.71f;
			var norms = new[] { (-1, 0) , (1, 0), (0, -0.6f), (0, 0.6f) };

			foreach (var normal in norms)
			{
				var vec = new Vector3(normal.Item1, normal.Item2);

				pln.Position = -vec;
				pln.Normal = vec;
				_sim.Spawn(pln);
			}
		}

		foreach (var elem in _sim.Enumerable)
		{
			if (elem is PhysSphere)
			{
				var spawned = Instantiate(SphereTemplate, Anchor);
				spawned.transform.localPosition = elem.Position;
				spawned.transform.localScale = Vector3.one * ((PhysSphere)elem).Radius * 2.0f;
				_mapping[elem] = spawned;
			}
			else if (elem is PhysStaticPlane)
			{
				var spawned = Instantiate(PlaneTemplate, Anchor);
				var tr = spawned.transform;
				tr.localPosition = elem.Position;
				tr.localRotation = Quaternion.LookRotation(((PhysStaticPlane)elem).Normal);
				_mapping[elem] = spawned;
			}
		}
	}

	// Update is called once per frame
	void Update()
	{
		if (Input.GetKeyDown(KeyCode.Tab))
		{
			_manualSimMode = !_manualSimMode;
		}

		bool bSimulate = !_manualSimMode || Input.GetKeyDown(KeyCode.Space);
		float simStep = _manualSimMode ? SimulationStep : Time.deltaTime;

		if (bSimulate)
		{
			var results = new List<PhysContext.ContactInfo>();
			_sim.StepSimulation(simStep, results);

			foreach (var elem in _sim.Enumerable)
			{
				_mapping[elem].transform.localPosition = elem.Position;
			}
		}
	}
}
