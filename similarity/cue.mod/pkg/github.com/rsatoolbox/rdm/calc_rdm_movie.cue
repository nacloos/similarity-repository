package rdm

#calc_rdm_movie: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Dataset
	dataset: _

	// Method
	method?: _ | *"euclidean"

	// Descriptor
	descriptor?: _ | *null

	// Noise
	noise?: _ | *null

	// Cv Descriptor
	cv_descriptor?: _ | *null

	// Prior Lambda
	prior_lambda?: _ | *1

	// Prior Weight
	prior_weight?: _ | *0.1

	// Time Descriptor
	time_descriptor?: _ | *"time"

	// Bins
	bins?: _ | *null

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.calc_rdm_movie"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
