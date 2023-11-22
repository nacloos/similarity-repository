package metrics

#GaussianStochasticMetric: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Alpha
	alpha?: number | *1.0

	// Group
	group?: "orth" | "perm" | "identity" | *"orth"

	// Init
	init?: "means" | "rand" | *"means"

	// Niter
	niter?: int | *1000

	// Tol
	tol?: number | *1e-8

	// Random State
	random_state?: (int | null) & null | *null

	// N Restarts
	n_restarts?: int | *1

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"netrep.metrics.GaussianStochasticMetric"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
