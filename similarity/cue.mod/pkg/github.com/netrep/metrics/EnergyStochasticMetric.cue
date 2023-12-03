package measures

#EnergyStochasticmeasure: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Group
	group?: "orth" | "perm" | "identity" | *"orth"

	// Niter
	niter?: int | *100

	// Tol
	tol?: number | *0.000001

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"netrep.measures.EnergyStochasticmeasure"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
