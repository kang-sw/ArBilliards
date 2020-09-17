using System.Collections;
using System.Collections.Generic;
using ArBilliards.Phys;
using UnityEngine;

public class BilliardsSimulationTest : MonoBehaviour
{
	public GameObject SphereTemplate;
	public Transform Anchor;
	public float SimulationStep = 0.5f;

	private PhysContext _sim = new PhysContext();
	private Dictionary<PhysObject, GameObject> _mapping = new Dictionary<PhysObject, GameObject>();
	private bool _manualSimMode = true;

	// Start is called before the first frame update
	void Start()
	{
		var obj = new PhysSphere();
		obj.DampingCoeff = 0.14f;
		obj.RestitutionCoeff = 0.8f;
		obj.Radius = 0.05f;
		obj.Velocity = new Vector3(0.2f, 0.0f, 0.0f);
		_sim.Spawn(obj);

		obj.Radius = 0.6f;
		obj.Mass = 2;
		obj.Position = new Vector3(0.8f, 0.0f, 0.0f);
		obj.Velocity = Vector3.zero;
		_sim.Spawn(obj);

		obj.Mass = 1f;
		obj.Radius = 0.06f;
		obj.Position = new Vector3(-0.3f, 0.1f, 0.0f);
		obj.Velocity = new Vector3(0.12f, -0.02f, 0.0f);
		_sim.Spawn(obj);

		foreach (var elem in _sim.Enumerable)
		{
			// elem.DampingCoeff = 0.1f;
			var spawned = Instantiate(SphereTemplate, Anchor);
			spawned.transform.localPosition = elem.Position;
			spawned.transform.localScale = Vector3.one * ((PhysSphere)elem).Radius * 2.0f;
			_mapping[elem] = spawned;
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
			var contacts = _sim.StepSimulation(simStep);

			foreach (var elem in _sim.Enumerable)
			{
				_mapping[elem].transform.localPosition = elem.Position;
			}
		}
	}
}
