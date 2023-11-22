package metrics

#LinearCKA: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Center Columns
	center_columns?: _ | *true

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"netrep.metrics.LinearCKA"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
