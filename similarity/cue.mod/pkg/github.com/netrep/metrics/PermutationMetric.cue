package measures

#Permutationmeasure: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Center Columns
	center_columns?: bool | *true

	// Zero Pad
	zero_pad?: bool | *true

	// Score Method
	score_method?: "angular" | "euclidean" | *"angular"

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"netrep.measures.Permutationmeasure"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
