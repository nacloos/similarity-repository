package rdm

#RDMs: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Dissimilarities
	dissimilarities: _

	// Dissimilarity Measure
	dissimilarity_measure?: _ | *null

	// Descriptors
	descriptors?: _ | *null

	// Rdm Descriptors
	rdm_descriptors?: _ | *null

	// Pattern Descriptors
	pattern_descriptors?: _ | *null

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.RDMs"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
