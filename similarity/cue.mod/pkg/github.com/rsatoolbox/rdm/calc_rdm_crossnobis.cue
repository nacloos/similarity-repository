package rdm

#calc_rdm_crossnobis: {
	@jsonschema(schema="http://json-schema.org/draft/2019-09/schema#")

	// Dataset
	dataset: _

	// Descriptor
	descriptor: _

	// Noise
	noise?: _ | *null

	// Cv Descriptor
	cv_descriptor?: _ | *null

	// Path to locate the object (the package has to be installed)
	"_target_": string | *"rsatoolbox.rdm.calc_rdm_crossnobis"
	"_wrap_"?: {
		...
	}
	"_out_"?: {
		...
	}
}
