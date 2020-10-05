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
	private Dictionary<PhysObject, GameObject> _mapping;
	private bool _manualSimMode = true;
	private float _counter = 0.0f;

	private Dictionary<PhysObject, List<Vector3>> _trails;

	// Start is called before the first frame update
	void Start()
	{
		Reset();
	}

	private void Reset()
	{
		if (_mapping != null)
		{
			foreach (var pair in _mapping)
			{
				Destroy(pair.Value);
			}
		}

		_mapping = new Dictionary<PhysObject, GameObject>();
		_trails = new Dictionary<PhysObject, List<Vector3>>();
		_sim.Clear();

		{
			var pln = new PhysStaticPlane();
			pln.Restitution = 0.71f;
			var norms = new[] { (-1f, 0), (1f, 0), (0f, -0.6f), (0, 0.6f), (-1.2f, 0), (1.2f, 0), (0, -0.8f), (0, 0.8f) };

			foreach (var normal in norms)
			{
				var vec = new Vector3(normal.Item1, normal.Item2);

				pln.SourcePosition = -vec;
				pln.Normal = vec;
				_sim.Spawn(pln);
			}
		}
		{
			var sph = new PhysSphere();
			sph.Mass = 10f;
			sph.Damping = 0.44;
			sph.Restitution = 0.65f;
			sph.Friction.Static = 0.12f;

			sph.Radius = 0.11f;
			sph.SourceVelocity = new Vector3(2.4f, 0.3f);
			_sim.Spawn(sph);

			sph.SourcePosition = new Vector3(0.96f, 0.0f);
			sph.SourceVelocity = Vector3.zero;
			_sim.Spawn(sph);

			sph.SourcePosition = new Vector3(0.54f, -0.15f);
			_sim.Spawn(sph);
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

		if (Input.GetKeyDown(KeyCode.Q))
		{
			Reset();
		}

		if (Input.GetKeyDown(KeyCode.Tab))
		{
			_manualSimMode = !_manualSimMode;
		}

		_counter += Time.deltaTime;
		bool bSimulate = (!_manualSimMode && _counter > SimulationStep) || Input.GetKeyDown(KeyCode.Space);

		if (bSimulate)
		{
			_counter = 0f;

			foreach (var obj in _sim.Enumerable)
			{
				if (!_trails.ContainsKey(obj))
				{
					_trails[obj] = new List<Vector3>();
					_trails[obj].Add(obj.Position);
				}
			}

			var contact = new List<PhysContext.ContactInfo>();
			_sim.StepSimulation(SimulationStep, contact);

			foreach (var elem in _sim.Enumerable)
			{
				_mapping[elem].transform.localPosition = elem.Position;
			}

			foreach (var elem in contact)
			{
				foreach (var A in new[] { elem.A, elem.B })
				{
					var obj = _sim[A.Idx];
					_trails[obj].Add(A.Pos);
				}
			}

			foreach (var obj in _sim.Enumerable)
			{
				_trails[obj].Add(obj.Position);
			}
		}

		foreach (var pair in _trails)
		{
			var joints = pair.Value;
			for (int i = 0; i < joints.Count - 1; i++)
			{
				var beg = joints[i] + Anchor.transform.position;
				var end = joints[i + 1] + Anchor.transform.position;

				Debug.DrawLine(beg, end, Color.red);
			}
		}
	}
}
